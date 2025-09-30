import os
import sys
import argparse
import logging
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

# Local imports
from diffusion_process import CE_DDIM_samplier
from model_diffusion import UNetDualHead_clean
from evaluate_metrics import (
    calculate_error_map, 
    calculate_batch_psnr, 
    calculate_batch_ssim, 
    calculate_batch_nmse, 
    create_water_baseline,
    compute_batch_crop_size
)
from dataset import create_paired_datasets

class EvalConfig:
    """Configuration class for evaluation parameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Model parameters
        self.T = 1000
        self.ch = 128
        self.ch_mult = [1, 2, 3, 4]
        self.attn = [2]
        self.num_res_blocks = 2
        self.dropout = 0.1
        self.input_channel = 4
        self.output_channel = 2
        self.num_channel = 2
        
        # Sampling parameters
        self.sample_steps = 10
        self.skip_steps = 1
        self.eta = 1.0
        self.noise_weight = 0.3
        self.refine_hu_thresh = 30
        self.refine_steps = 0
        self.noise_scheduler = 'linear'
        
        # Data parameters
        self.image_size = 256
        self.batch_size = 5
        self.vis_channel = 0  # Channel to visualize (0-based)
        
        # Diffusion parameters
        self.beta_1 = 4e-4
        self.beta_T = 0.02
        
        # HU range for medical imaging
        self.hu_min = -1000
        self.hu_max = 3000
    
        # Paths (to be set via arguments)
        self.data_path = ""
        self.model_path = ""
        self.output_dir = ""
        
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


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('dual_head_evaluation')
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(log_dir, f'evaluation_{experiment_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dual-Head Diffusion Model Evaluation')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to evaluation data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, help='Experiment name for logging')
    parser.add_argument('--batch_size', type=int, help='Batch size for evaluation')
    parser.add_argument('--sample_steps', type=int, help='Number of sampling steps')
    
    args = parser.parse_args()
    
    # Load configuration
    config = EvalConfig(args.config)
    config.data_path = args.data_path
    config.model_path = args.model_path
    config.output_dir = args.output_dir
    
    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.sample_steps:
        config.sample_steps = args.sample_steps
    if args.refine_steps is not None:
        config.refine_steps = args.refine_steps
    
    # Setup experiment name
    if not args.experiment_name:
        model_name = Path(args.model_path).stem
        args.experiment_name = f"eval_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory
    experiment_dir = os.path.join(config.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(experiment_dir, args.experiment_name)
    
    logger.info("=" * 80)
    logger.info("DUAL-HEAD DIFFUSION MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Data: {config.data_path}")
    logger.info(f"Output: {experiment_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset
    logger.info("Loading dataset...")
    _, test_dataset = create_paired_datasets(
        data_dir=config.data_path, 
        split_ratio=0.8, 
        image_size=config.image_size
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Test batches: {len(test_dataloader)}")
    
    # Load model
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {config.model_path}")
    
    logger.info(f"Loading model from {config.model_path}")
    
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
    
    # Load checkpoint
    checkpoint = torch.load(config.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded checkpoint (epoch information not available)")
    
    model.eval()
    
    # Create sampler
    sampler = CE_DDIM_samplier(
        model=model,
        sample_steps=config.sample_steps,
        skip_steps=config.skip_steps,
        eta=config.eta,
        num_channel=config.num_channel,
        alpha_star=args.alpha_star,
        hu_range=(config.hu_min, config.hu_max),
        sigma_thresh_hu=config.refine_hu_thresh
    ).to(device)
    
    # Evaluation loop
    logger.info("Starting evaluation...")
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_nmse = 0.0
    total_inference_time = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            start_time = time.time()
            
            # Prepare data
            ct = batch["pct"].to(device) * batch["mask"].to(device)
            cbct = batch["cbct"].to(device) * batch["mask"].to(device)
            mask = batch["mask"].to(device)
            
            # Generate noise for sampling
            noise = torch.randn_like(cbct).to(device)
            x_T = torch.cat((noise, cbct), dim=1)
            
            # Reconstruct with uncertainty
            mu_norm, _, _, _ = sampler.reconstruct(
                x_T=x_T, refine=config.refine_steps > 0
            )
            
            batch_time = time.time() - start_time
            total_inference_time += batch_time
            
            # Compute metrics
            batch_psnr = calculate_batch_psnr(
                ct.cpu().numpy(), mu_norm.cpu().numpy(), 
                mask.cpu().numpy(), channel=config.vis_channel
            )
            batch_ssim = calculate_batch_ssim(
                ct.cpu().numpy(), mu_norm.cpu().numpy(), 
                mask.cpu().numpy(), channel=config.vis_channel
            )
            batch_nmse = calculate_batch_nmse(
                ct.cpu().numpy(), mu_norm.cpu().numpy(), 
                mask.cpu().numpy(), channel=config.vis_channel
            )
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_nmse += batch_nmse
            
            # Log progress
            if batch_idx % 10 == 0 or batch_idx == len(test_dataloader) - 1:
                logger.info(f"Batch {batch_idx + 1}/{len(test_dataloader)} - "
                           f"PSNR: {batch_psnr:.2f}dB, SSIM: {batch_ssim:.4f}, "
                           f"NMSE: {batch_nmse:.4f}, Time: {batch_time:.2f}s")
    
    # Compute averages
    num_batches = len(test_dataloader)
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_nmse = total_nmse / num_batches
    avg_time = total_inference_time / num_batches
    
    # Log final results
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")
    logger.info(f"Average NMSE: {avg_nmse:.4f}")
    logger.info(f"Average inference time per batch: {avg_time:.2f}s")
    logger.info(f"Average inference time per sample: {avg_time/config.batch_size:.2f}s")
    logger.info("=" * 80)
    
    # Save results
    results = {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_nmse': avg_nmse,
        'avg_inference_time_per_batch': avg_time,
        'avg_inference_time_per_sample': avg_time / config.batch_size,
        'config': config.__dict__
    }
    
    results_file = os.path.join(experiment_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()

