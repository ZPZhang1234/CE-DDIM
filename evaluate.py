import os
from typing import Dict
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from diffusion_process import CE_DDIM_samplier
from model_diffusion import UNetDualHead_clean
from evaluate_metrics import calculate_error_map, calculate_batch_psnr, calculate_batch_ssim, calculate_batch_nmse, create_water_baseline,compute_batch_crop_size
from dataset import create_paired_datasets
import torch.optim.lr_scheduler as lr_scheduler
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk
import torch.cuda.amp as amp  # Import Automatic Mixed Precision
from IPython import display
# Training parameterss
dataset_name = "cbct"
image_size = 256
batch_size = 5
T = 1000
ch = 128
ch_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.1
beta_1 = 4e-4
beta_T = 0.02
sample_steps = 10
con_weight = 1
eta = 1
skip_steps = 1
input_channel = 4
output_channel = 2
vis_channel = 0
num_channel = 2
noise_weight = 0.3
clip_range = 50
refin_hu_thresh = 30
refine_steps = 0
noise_scheduler = 'linear'
out_name = f""
output_dir = ""%out_name
data_path = ''
print("out_name:",out_name)

def fit_alpha_star(val_loader, sampler, device, mask_key="mask"):
    ratios = []
    with torch.no_grad():
        for batch in val_loader:
            ct = batch["pct"].to(device) * batch[mask_key].to(device)
            cbct = batch["cbct"].to(device) * batch[mask_key].to(device)
            mask = batch["mask"].to(device).bool()
            noise = torch.randn_like(cbct)
            x_T = torch.cat((noise, cbct), dim=1) 
            mu_norm, sigma_norm, mu_hu, sigma_hu = sampler.reconstruct(x_T=x_T, refine=False)
            err = (ct - mu_hu).abs()
            r = (err / (sigma_hu + 1e-8))[mask.expand_as(err)]
            ratios.append(r.flatten().cpu().numpy())
    ratios = np.concatenate(ratios)
    return np.quantile(ratios, 0.95) 


@torch.no_grad()
def coverage_at_levels(mu_hu, sigma_hu, ct, mask, levels=(0.683, 0.95, 0.99), alpha_star=1.0):
    device = mu_hu.device
    mask   = mask.to(device).to(torch.bool)

    err = (ct.to(device) - mu_hu).abs()
    sig = (sigma_hu.to(device) * alpha_star).clamp_min(1e-8)

    z = torch.masked_select(err / sig, mask.expand_as(err))
    if z.numel() == 0:
        return {p: float("nan") for p in levels}

    kmap = {0.683: 1.0, 0.95: 1.96, 0.99: 2.576}
    cov = {}
    for p in levels:
        k = kmap.get(p, torch.distributions.Normal(0,1).icdf(torch.tensor((1+p)/2.0)).item())
        cov[p] = float((z <= k).float().mean().cpu())
    return cov

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
maeloss = torch.nn.L1Loss()
# Create checkpoints directory
os.makedirs("%s" % output_dir, exist_ok=True)

# Create training and testing datasets and loaders
_, test_dataset = create_paired_datasets(data_dir=data_path, split_ratio=0.8, image_size = image_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  # Increase to 4 workers
# Load the model used by both DDIM and DDPM samplers
net_model = UNetDualHead_clean(
    T=T,
    input_channel=input_channel,
    output_channel = output_channel,
    ch=ch,
    ch_mult=ch_mult,
    attn=attn,
    num_res_blocks=num_res_blocks,
    dropout=dropout
).to(device)

net_model.eval()

sampler = CE_DDIM_samplier(model = net_model, sample_steps=sample_steps,
                 skip_steps=skip_steps, eta=eta,
                 num_channel=num_channel, alpha_star=1.0,
                 hu_range=(-1000, 3000),
                 sigma_thresh_hu=refin_hu_thresh,  # HU
                 refine_steps=refine_steps, refine_tile=32).to(device)
start_epoch = 1
net_model_ckpt_path = ''

if os.path.exists(net_model_ckpt_path):
    print(f"Loading checkpoint from {net_model_ckpt_path}...")
    checkpoint = torch.load(net_model_ckpt_path)
    net_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Resuming from epoch {start_epoch}")

alpha_star = 1.0

with torch.no_grad():
    net_model.eval()
    batch_psnr_ddim_total = 0
    batch_ssim_ddim_total = 0
    batch_nmse_ddim_total = 0
    print("evaluation process starts")
    for i, batch in enumerate(test_dataloader):

        start_batch_time = time.time()  
        ct = batch["pct"].to(device) * batch["mask"].to(device) 
        cbct = batch["cbct"].to(device) * batch["mask"].to(device)  
        mask = batch["mask"].to(device) 
        noise = torch.randn_like(cbct).to(device) 
        x_T = torch.cat((noise, cbct), dim=1)   
        mu_norm, sigma_norm, mu_HU, sigma_HU = sampler.reconstruct(x_T=x_T,refine=True) 
        x_0_sampled_ddim = mu_norm

        ddim_inference_time_point = time.time()
        ddim_inference_time = ddim_inference_time_point - start_batch_time
        print(f"Batch {i} DDIM Inference Time: {ddim_inference_time:.2f} seconds")




        batch_psnr_ddim = calculate_batch_psnr(ct.cpu().numpy(), x_0_sampled_ddim.cpu().numpy(), mask.cpu().numpy(),channel=vis_channel)
        batch_ssim_ddim = calculate_batch_ssim(ct.cpu().numpy(), x_0_sampled_ddim.cpu().numpy(), mask.cpu().numpy(),channel=vis_channel)
        batch_nmse_ddim = calculate_batch_nmse(ct.cpu().numpy(), x_0_sampled_ddim.cpu().numpy(), mask.cpu().numpy(),channel=vis_channel)


        batch_psnr_ddim_total += batch_psnr_ddim
        batch_ssim_ddim_total += batch_ssim_ddim
        batch_nmse_ddim_total += batch_nmse_ddim

        # Print batch results
        print(f"Batch {i} - Batch PSNR (Predicted sCT DDIM vs Ground Truth): {batch_psnr_ddim:.2f} dB", flush=True)
        print(f"Batch {i} - Batch SSIM (Predicted sCT DDIM vs Ground Truth): {batch_ssim_ddim:.4f}", flush=True)
        print(f"Batch {i} - Batch NMSE (Predicted sCT DDIM vs Ground Truth): {batch_nmse_ddim:.4f}", flush=True)

        # Calculate and print inference time for each batch
        end_batch_time = time.time()
        batch_inference_time = end_batch_time - start_batch_time
        plotting_time = end_batch_time - ddim_inference_time_point
        print(f"Batch {i} Plotting Time: {plotting_time:.2f} seconds")
        print(f"Batch {i} Inference Time: {batch_inference_time:.2f} seconds")

    num_batches = len(test_dataloader)

    # Averages for DDIM model
    avg_psnr_ddim = batch_psnr_ddim_total / num_batches
    avg_ssim_ddim = batch_ssim_ddim_total / num_batches
    avg_nmse_ddim = batch_nmse_ddim_total / num_batches

    # Print overall averages
    print("Average Metrics over all batches:")
    print(f"Avg PSNR (Predicted sCT DDIM vs Ground Truth): {avg_psnr_ddim:.2f} dB", flush=True)
    print(f"Avg SSIM (Predicted sCT DDIM vs Ground Truth): {avg_ssim_ddim:.4f}", flush=True)
    print(f"Avg NMSE (Predicted sCT DDIM vs Ground Truth): {avg_nmse_ddim:.4f}", flush=True)

