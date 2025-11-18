# CE-DDIM: Conditional Efficient Diffusion Method for Cone-Beam CT Enhancement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A PyTorch implementation of a dual-head diffusion model that provides both high-quality image denoising and calibrated uncertainty estimates. This model architecture features separate heads for noise prediction (eps_head) and uncertainty estimation (logvar_head), enabling uncertainty-aware image reconstruction.


## 📊 Usage

### Training

**Basic training:**
```bash
python train.py --data_path /path/to/training/data --pretrained_path /path/to/single_head_model.pt
```

**Brain dataset training:**
```bash
python train.py --data_path /path/to/brain/data --dataset_type brain --pretrained_path /path/to/model.pt
```

**Multi-GPU distributed training:**
```bash
torchrun --nproc_per_node=4 train.py --data_path /path/to/data --dataset_type pelvis --use_wandb --experiment_name my_experiment
```

### Evaluation

**Basic evaluation:**
```bash
python evaluate.py --model_path /path/to/trained_model.pt --data_path /path/to/test/data
```

**Brain dataset evaluation:**
```bash
python evaluate.py --model_path /path/to/brain_model.pt --data_path /path/to/brain/data --dataset_type brain
```

**With custom configuration:**
```bash
python evaluate.py --config configs/default_eval.yaml --model_path /path/to/model.pt --data_path /path/to/data --dataset_type brain
```

**Custom evaluation settings:**
```bash
python evaluate.py --model_path /path/to/model.pt --data_path /path/to/data --dataset_type pelvis --sample_steps 20 --batch_size 10
```

## 📁 Project Structure

```
dual-head-diffusion/
├── train.py                 # Training script
├── evaluate.py             # Evaluation script
├── configs/
│   ├── default_train.yaml  # Default training configuration
│   └── default_eval.yaml   # Default evaluation configuration
├── diffusion_process.py    # Diffusion training and sampling logic
├── model_diffusion.py      # Model architectures
├── dataset.py              # Dataset loading utilities
├── evaluate_metrics.py     # Evaluation metrics
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🏗️ Model Architecture

The dual-head diffusion model consists of:

1. **Shared Backbone**: UNet encoder-decoder for feature extraction
2. **Epsilon Head (eps_head)**: Predicts noise for denoising 
3. **Log-variance Head (logvar_head)**: Predicts log-variance estimates 

<!-- ### Training Strategy

1. **Pre-training**: Train single-head model for noise prediction
2. **Dual-head initialization**: Load pre-trained weights and add log-variance head
3. **log-variance training**: Freeze eps_head, train backbone + logvar_head with log-variance loss -->

## 📈 Key Parameters

### Dataset Configuration

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `dataset_type` | Medical imaging dataset type | `pelvis`, `brain` | `pelvis` |
| `image_size` | Input image resolution | Any integer | 256 |
| `batch_size` | Training batch size | Any integer | 50 |

**Dataset Types:**
- **Pelvis**: Uses bone (1000/4000), soft tissue (50/400), and intermediate (600/3000) windowing
- **Brain**: Uses bone (1000/4000), soft tissue (50/400), and brain-specific (35/80) windowing

### Training Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `backbone_lr` | Learning rate for backbone | 1e-5 |
| `logvar_lr` | Learning rate for uncertainty head | 1e-4 |
| `LAM_NLL_BASE` | NLL loss weight | 5.0 |
| `n_epochs` | Number of training epochs | 50000 |

### Evaluation Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sample_steps` | DDIM sampling steps | 5 |
| `alpha_star` | Uncertainty calibration factor | 1.0 |

## 🔬 Evaluation Metrics

The evaluation script computes:

- **Image Quality**: PSNR, SSIM, NMSE
- **Timing**: Inference time per batch and per sample

## 🎯 Results

Expected performance on medical CT denoising:
- **PSNR**: 35-38 dB
- **SSIM**: 0.90-0.95
- **Inference Time**: ~0.5-2s per sample (depending on sampling steps)

## 🔧 Advanced Usage

### Custom Loss Functions

The training script uses a combination of:
- **L_var**: Negative log-likelihood for uncertainty learning
- **TV Loss**: Total variation regularization for spatial smoothness
- **Prior Loss**: L2 regularization toward expected noise levels
<!-- 
### Uncertainty Calibration

The `alpha_star` parameter scales predicted uncertainties for proper calibration:
```python
# Fit alpha_star automatically
python evaluate.py --fit_alpha_star --model_path model.pt --data_path data/

# Use specific alpha_star value
python evaluate.py --alpha_star 1.2 --model_path model.pt --data_path data/
``` -->

<!-- ## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Dual-Head Diffusion Models for Uncertainty-Aware Image Denoising},
  author={Your Name and Co-authors},
  journal={Your Journal},
  year={2024}
}
``` -->

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: zipei.zhang@aalto.fi