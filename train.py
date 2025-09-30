
import os
import sys
import argparse
import logging
import time
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler

# Third-party imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be limited.")

from zclip import ZClip
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

# Local imports
from diffusion_process import CE_DDIM_trainer
from model_diffusion import UNetDualHead_clean
from dataset import create_paired_datasets


class Config:
    """Configuration class for training parameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration
        self.dataset_name = "cbct"
        self.dataset_type = "pelvis"  # pelvis or brain
        self.noise_scheduler = 'linear'
        self.image_size = 256
        self.batch_size = 50
        self.T = 1000
        self.ch = 128
        self.ch_mult = [1, 2, 3, 4]
        self.attn = [2]
        self.num_res_blocks = 2
        self.dropout = 0.2
        self.n_epochs = 50000
        self.beta_1 = 4e-4
        self.beta_T = 0.02
        self.save_interval = 200
        self.test_interval = 20000
        self.sample_steps = 400
        self.input_channel = 6
        self.output_channel = 3
        self.num_channel = 3
        
        # Learning rates
        self.backbone_lr = 1e-5
        self.logvar_lr = 1e-4
        
        # Loss parameters
        self.LAM_NLL_BASE = 5
        self.N_WARM = 0
        
        # Paths (to be set via arguments)
        self.data_path = ""
        self.save_weight_dir = ""
        self.output_dir = ""
        self.pretrained_path = ""
        
        # Logging
        self.use_wandb = False
        self.project_name = "dual-head-diffusion"
        self.experiment_name = None
        
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_yaml(self, save_path: str):
        """Save configuration to YAML file."""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('dual_head_training')
    logger.setLevel(logging.INFO)
    
    # Only log from rank 0 in distributed training
    if rank == 0:
        # File handler
        log_file = os.path.join(log_dir, f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        return rank, world_size, local_rank, device
    else:
        # Single GPU training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device


def load_pretrained_weights(model: nn.Module, pretrained_path: str, logger: logging.Logger):
    """Load pretrained single-head weights into dual-head model."""
    if not os.path.exists(pretrained_path):
        logger.warning(f"Pretrained weights not found at {pretrained_path}. Using random initialization.")
        return
    
    logger.info(f"Loading pretrained weights from {pretrained_path}")
    
    ckpt = torch.load(pretrained_path, map_location="cpu")
    single_sd = ckpt.get("model_state_dict", ckpt)
    
    dual_sd = model.state_dict()
    loaded_keys = []
    
    for k, v in single_sd.items():
        if k.startswith("tail.2"):  # conv weights/bias
            new_k = k.replace("tail.2", "eps_head")
        elif k.startswith("tail.0"):  # group-norm
            new_k = k.replace("tail.0", "norm_act.0")
        else:
            new_k = k  # backbone: names identical
        
        if new_k in dual_sd and dual_sd[new_k].shape == v.shape:
            dual_sd[new_k] = v.clone()
            loaded_keys.append(new_k)
    
    model.load_state_dict(dual_sd, strict=False)
    logger.info(f"Successfully loaded {len(loaded_keys)} parameter groups from pretrained model")


def setup_model_and_optimizer(config: Config, device: torch.device, logger: logging.Logger) -> Tuple[nn.Module, optim.Optimizer]:
    """Setup model and optimizer."""
    # Create model
    model = UNetDualHead_clean(
        T=config.T,
        input_channel=config.input_channel,
        output_channel=config.output_channel,
        ch=config.ch,
        ch_mult=config.ch_mult,
        attn=config.attn,
        num_res_blocks=config.num_res_blocks,
        dropout=config.dropout
    ).to(device)
    
    # Load pretrained weights if available
    if config.pretrained_path:
        load_pretrained_weights(model, config.pretrained_path, logger)
    
    # Setup parameter groups
    eps_params = []
    logvar_params = []
    
    for name, param in model.named_parameters():
        if 'logvar_head' in name:
            logvar_params.append(param)
        elif 'eps_head' in name:
            param.requires_grad_(False)  # Freeze eps_head
        else:
            eps_params.append(param)  # Trainable backbone
    
    # Create optimizer
    optimizer = torch.optim.AdamW([
        {'params': eps_params, 'lr': config.backbone_lr},
        {'params': logvar_params, 'lr': config.logvar_lr}
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} total parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    
    return model, optimizer


def lam_nll(epoch: int, N_WARM: int, LAM_NLL_BASE: float) -> float:
    """Compute lambda parameter for NLL loss scheduling."""
    if epoch <= N_WARM:
        return 0.0
    ramp = min(1.0, (epoch - N_WARM) / 100)
    return LAM_NLL_BASE * ramp


def train_epoch(model: nn.Module, trainer: nn.Module, dataloader: DataLoader, 
                optimizer: optim.Optimizer, epoch: int, config: Config, 
                device: torch.device, logger: logging.Logger) -> Dict[str, float]:
    model.train()  
    # Metrics tracking
    metrics = {
        'loss': 0.0,
        'L_eps': 0.0,
        'L_var': 0.0,
        'tv_loss': 0.0,
        'prior_loss': 0.0,
        'count': 0
    }
    
    # Gradient clipping
    zclipping = ZClip(alpha=0.97, z_thresh=5)
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Prepare data
        ct = batch["pct"].to(device) * batch["mask"].to(device)
        cbct = batch["cbct"].to(device) * batch["mask"].to(device)
        concatenated_tensor = torch.cat((ct, cbct), dim=1)
        
        # Forward pass
        lam_nll_param = lam_nll(epoch, config.N_WARM, config.LAM_NLL_BASE)
        L_eps, L_var, tv_loss, prior_loss = trainer.forward(concatenated_tensor, lam_nll_param)
        
        # Compute final loss (only variance-related terms)
        loss = L_var + tv_loss + prior_loss
        
        # Backward pass
        loss.backward()
        zclipping.step(model)
        optimizer.step()
        
        # Update metrics
        metrics['loss'] += loss.item()
        metrics['L_eps'] += L_eps.item()
        metrics['L_var'] += L_var.item()
        metrics['tv_loss'] += tv_loss.item()
        metrics['prior_loss'] += prior_loss.item()
        metrics['count'] += 1
        
        # Log batch progress
        if batch_idx % 100 == 0 and dist.get_rank() == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                       f"Loss: {loss.item():.6f}")
    
    # Average metrics
    for key in metrics:
        if key != 'count':
            metrics[key] /= metrics['count']
    
    return metrics


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                   loss: float, save_path: str, logger: logging.Logger):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dual-Head Diffusion Model Training')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--dataset_type', type=str, choices=['pelvis', 'brain'], default='pelvis', 
                       help='Dataset type: pelvis or brain (affects windowing parameters)')
    parser.add_argument('--pretrained_path', type=str, help='Path to pretrained single-head model')
    parser.add_argument('--save_dir', type=str, default='./experiments', help='Directory to save outputs')
    parser.add_argument('--experiment_name', type=str, help='Experiment name for logging')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    
    # Load configuration
    config = Config(args.config)
    config.data_path = args.data_path
    config.dataset_type = args.dataset_type
    if args.pretrained_path:
        config.pretrained_path = args.pretrained_path
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.use_wandb:
        config.use_wandb = args.use_wandb
    
    # Setup experiment directory
    if not config.experiment_name:
        config.experiment_name = f"dual_head_{config.dataset_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiment_dir = os.path.join(args.save_dir, config.experiment_name)
    config.save_weight_dir = os.path.join(experiment_dir, 'checkpoints')
    config.output_dir = os.path.join(experiment_dir, 'outputs')
    
    # Create directories
    if rank == 0:
        os.makedirs(config.save_weight_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Save configuration
        config.save_to_yaml(os.path.join(experiment_dir, 'config.yaml'))
    
    # Setup logging
    logger = setup_logging(experiment_dir, rank)
    
    if rank == 0:
        logger.info("="*80)
        logger.info("DUAL-HEAD DIFFUSION MODEL TRAINING")
        logger.info("="*80)
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Dataset type: {config.dataset_type}")
        logger.info(f"Device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Data path: {config.data_path}")
        logger.info(f"Save directory: {experiment_dir}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    
    # Create datasets
    train_set, test_dataset = create_paired_datasets(
        data_dir=config.data_path, 
        split_ratio=0.8, 
        image_size=config.image_size,
        dataset_type=config.dataset_type
    )
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=2,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
    )
    
    if rank == 0:
        logger.info(f"Training samples: {len(train_set)}")
        logger.info(f"Test samples: {len(test_dataset)}")
    
    # Setup model and optimizer
    model, optimizer = setup_model_and_optimizer(config, device, logger)
    
    # Wrap model with DDP for distributed training
    if world_size > 1:
        dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create trainer
    trainer = CE_DDIM_trainer(
        model=model,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        T=config.T,
        num_channel=config.num_channel,
        noise_scheduler=config.noise_scheduler
    ).to(device)
    
    # Create scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=100
    )
    
    # Initialize Weights & Biases
    if config.use_wandb and WANDB_AVAILABLE and rank == 0:
        wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            config=config.__dict__
        )
        wandb.watch(model, log="all", log_freq=50)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            logger.info(f"Resumed from checkpoint: {args.resume}, epoch {start_epoch}")
    
    # Training loop
    if rank == 0:
        logger.info("Starting training...")
    
    for epoch in range(start_epoch, config.n_epochs + 1):
        start_time = time.time()
        
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        metrics = train_epoch(model, trainer, train_dataloader, optimizer, 
                            epoch, config, device, logger)
        
        # Learning rate scheduling
        scheduler.step(metrics['loss'])
        
        # Logging
        if rank == 0:
            epoch_time = time.time() - start_time
            epochs_left = config.n_epochs - epoch
            eta = datetime.timedelta(seconds=int(epochs_left * epoch_time))
            
            logger.info(f"Epoch [{epoch}/{config.n_epochs}] - "
                       f"Loss: {metrics['loss']:.6f}, "
                       f"L_var: {metrics['L_var']:.6f}, "
                       f"Time: {epoch_time:.2f}s, "
                       f"ETA: {eta}")
            
            # Weights & Biases logging
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": metrics['loss'],
                    "train/L_eps": metrics['L_eps'],
                    "train/L_var": metrics['L_var'],
                    "train/tv_loss": metrics['tv_loss'],
                    "train/prior_loss": metrics['prior_loss'],
                    "train/lr_backbone": optimizer.param_groups[0]['lr'],
                    "train/lr_logvar": optimizer.param_groups[1]['lr'],
                    "train/epoch_time": epoch_time
                })
        
        # Save checkpoint
        if epoch % config.save_interval == 0 and rank == 0:
            checkpoint_path = os.path.join(config.save_weight_dir, f"ckpt_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, metrics['loss'], 
                          checkpoint_path, logger)
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Cleanup
    if config.use_wandb and WANDB_AVAILABLE and rank == 0:
        wandb.finish()
    
    if world_size > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()