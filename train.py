import os
from zclip import ZClip
from typing import Dict
import time
import datetime
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from diffusion_process import  CE_DDIM_trainer
from model_diffusion import UNetDualHead,UNetDualHead_clean,UNet
from dataset import create_paired_datasets
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk
import torch.cuda.amp as amp  # Import Automatic Mixed Precision
from IPython import display
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler  
dataset_name = "cbct"
noise_scheduler = 'linear'
zclipping = ZClip(alpha=0.97, z_thresh=5)
image_size = 256
batch_size = 50
T = 1000
ch = 128
ch_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.2
lr = 1e-4
n_epochs = 50000
beta_1 = 4e-4
beta_T = 0.02
grad_clip = 2
save_interval = 200
test_interval = 20000
sample_steps = 400
con_weight = 1
eta = 1
skip_steps = 20
input_channel = 6
output_channel = 3
num_channel = 3
noise_weight = 0.1
clip_range = 20
save_weight_dir = f""
output_dir = f""
data_path = ""

LAM_NLL_BASE = 5
N_WARM = 0
        
def lam_nll(epoch):
    if epoch <= N_WARM:
        return 0.0
    ramp = min(1.0, (epoch - N_WARM) / 100)
    return LAM_NLL_BASE * ramp

dist.init_process_group(backend="nccl")         # For GPU-based training
local_rank = int(os.environ["LOCAL_RANK"])      # Local rank provided by torchrun
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
# For reproducibility, you might want to set the same seed on every rank:
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
# -----------------------------
# 3) Prepare Directories (rank 0 only)
# -----------------------------
if dist.get_rank() == 0:
    os.makedirs(save_weight_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

# 5) Create Datasets and Dataloaders
# -----------------------------
train_set, test_dataset = create_paired_datasets(
    data_dir=data_path, split_ratio=0.8, image_size=image_size
)
train_sampler = DistributedSampler(
    train_set,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True
)
test_sampler = DistributedSampler(
    test_dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=False
)

train_dataloader = DataLoader(
    train_set,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=2,
    pin_memory=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    sampler=test_sampler,
    num_workers=2,
    pin_memory=True,
)
# -----------------------------
# 6) Model, Optimizer, and Checkpoint
# -----------------------------
net_model = UNetDualHead_clean(
    T=T,
    input_channel=input_channel,
    output_channel=output_channel,
    ch=ch,
    ch_mult=ch_mult,
    attn=attn,
    num_res_blocks=num_res_blocks,
    dropout=dropout
).to(device)

net_model = net_model.to(device)
# ─────────────────────────────────────────────────────────────
# 2.  read single-head weights
# ─────────────────────────────────────────────────────────────
ckpt = torch.load("", map_location="cpu")
single_sd = ckpt["model_state_dict"]            # or ckpt if saved raw

# ─────────────────────────────────────────────────────────────
# 3.  map keys:   tail.0 → norm_act.0      (GroupNorm)
#                 tail.2 → eps_head        (Conv)
#                 (tail.1 is Swish → no params)
# ─────────────────────────────────────────────────────────────
dual_sd = net_model.state_dict()                     # current weights
for k, v in single_sd.items():
    if k.startswith("tail.2"):                  # conv weights / bias
        new_k = k.replace("tail.2", "eps_head")
    elif k.startswith("tail.0"):                # group-norm
        new_k = k.replace("tail.0", "norm_act.0")
    else:
        new_k = k                               # backbone: names identical
    if new_k in dual_sd and dual_sd[new_k].shape == v.shape:
        dual_sd[new_k] = v.clone()

# # OPTIONAL:  initialise the new log-variance head to log σ² ≃ −5
# nn.init.zeros_(dual.logvar_head.weight)
# nn.init.constant_(dual.logvar_head.bias, -5.0)

net_model.load_state_dict(dual_sd, strict=False)     # strict=False → ignore logvar_head

# ─────────────────────────────────────────────────────────────
# 4.  choose which parameters to train
#     (freeze everything except logvar_head by default)
# # ─────────────────────────────────────────────────────────────
# for name, p in net_model.named_parameters():
#     p.requires_grad_(name.startswith("logvar_head"))


print("Loaded single-head weights → dual-head model.")
# print("Trainable params:",
#       sum(p.numel() for p in net_model.parameters() if p.requires_grad))

# split parameters ---------------------------------------
eps_params    = []
logvar_params = []
for n, p in net_model.named_parameters():
    if 'logvar_head' in n:
        logvar_params.append(p)
    else:
        p.requires_grad_(False)   # ← freeze ε-head
        eps_params.append(p)

optimizer = torch.optim.AdamW(
    [
        {'params': eps_params,    'lr': 1e-5},   # backbone + ε-head
        {'params': logvar_params, 'lr': 1e-4}    # variance head (100× smaller)
    ],
    betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5
)

# opt_var = torch.optim.AdamW(
#         [p for p in net_model.parameters() if p.requires_grad],
#         lr=5e-5, betas=(0.9, 0.999))

start_epoch = 1
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     net_model.load_state_dict(checkpoint["model_state_dict"])
#     # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     start_epoch = checkpoint["epoch"] + 1
#     print(f"Resuming from epoch {start_epoch}")
# Synchronize all processes to ensure model weights are the same
dist.barrier()
# Wrap model with DDP
net_model = DDP(net_model, device_ids=[local_rank], output_device=local_rank)
# ----------------------------
# 7) Trainer, Sampler & Scheduler
# -----------------------------
trainer = CE_DDIM_trainer(
    model=net_model,
    beta_1=beta_1,
    beta_T=beta_T,
    T=T,
    num_channel=num_channel,
    noise_scheduler=noise_scheduler
).to(device)


scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,
    patience=100
)
# -----------------------------
# 8) Training Loop
# -----------------------------
# scaler = torch.amp.GradScaler('cuda',enabled=True)
prev_time = time.time()

def log_memory_usage():
    if dist.get_rank() == 0:
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"GPU Memory: Allocated {allocated:.2f}MB, Reserved {reserved:.2f}MB")

accumulation_steps = 1 
# Empty cache at regular intervals
def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

for epoch in range(start_epoch, n_epochs + 1):
    train_sampler.set_epoch(epoch)
    net_model.train()
    local_loss_sum = 0.0
    local_sample_count = 0  
    L_eps_sum = 0
    L_var_sum = 0
    tv_loss_sum = 0
    prior_loss_sum = 0
    # if epoch < 100:
    #     optimizer.param_groups[1]['lr'] = 0.0        # freeze variance head
    # else:
    #     optimizer.param_groups[1]['lr'] = 1e-7       # un-freeze with tiny LR
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        ct = batch["pct"].to(device) * batch["mask"].to(device) 
        cbct = batch["cbct"].to(device) * batch["mask"].to(device)  
        concancated_tensor = torch.cat((ct, cbct), dim=1)
        lam_nll_param = lam_nll(epoch)
        # with torch.autocast(device_type="cuda"):
        L_eps, L_var, tv_loss, prior_loss = trainer.forward(concancated_tensor, lam_nll_param)

        # --- final scalar loss --------------------------------------------------
        #   • During warm-up   (lam_nll_param==0) we train only the ε-head → loss=L_eps
        # #   • After warm-up    we add NLL + regularisers, weighted by lam_nll_param.
        # if lam_nll_param == 0.0:
        #     loss = L_eps                              # plain MSE, variance head frozen
        # else:
        #     loss = L_eps + lam_nll_param * (L_var + tv_loss + prior_loss)
        loss = L_var + tv_loss + prior_loss
        local_loss_sum += loss.item()
        L_eps_sum     += L_eps.item()
        L_var_sum    += L_var.item()
        tv_loss_sum  += tv_loss.item()
        prior_loss_sum += prior_loss.item()
        local_sample_count += 1

        loss.backward()
        zclipping.step(net_model)
        optimizer.step()
        # scaler.unscale_(optimizer)  # Unscale gradients before clipping
        # # torch.nn.utils.clip_grad_norm_(net_model.parameters(), grad_clip)
        # scaler.step(optimizer)
        # scaler.update()
    local_loss_tensor = torch.tensor([local_loss_sum, local_sample_count], 
                                     dtype=torch.float, device=device)
    dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
    global_loss_sum     = local_loss_tensor[0].item()
    global_sample_count = local_loss_tensor[1].item()
    avg_loss = 0.0
    L_eps_avg = L_eps_sum/local_sample_count
    L_var_avg = L_var_sum/local_sample_count
    tv_loss_avg = tv_loss_sum/local_sample_count
    prior_loss_avg = prior_loss_sum/local_sample_count

    if global_sample_count > 0:
        avg_loss = global_loss_sum / global_sample_count

    time_duration = datetime.timedelta(seconds=(time.time() - prev_time))
    epoch_left = n_epochs - epoch
    time_left = datetime.timedelta(seconds=epoch_left * (time.time() - prev_time))
    prev_time = time.time()

    if epoch > 2 and epoch % save_interval == 0 and dist.get_rank() == 0:
        ckpt_path = os.path.join(save_weight_dir, f"ckpt_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net_model.module.state_dict(), 
                "loss": avg_loss,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path
        )
        print(f"[Rank 0] Saved checkpoint: {ckpt_path}")

    sys.stdout.write(
        "\r[Epoch %d/%d] [ETA: %s] [EpochDuration: %s] [Global Loss: %s]"
        % (
            epoch,
            n_epochs,
            time_left,
            time_duration,
            avg_loss,
        )
    )
    empty_cache()