import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk
import torch

def calculate_error_map(pred_img, gt_img, mask):

    return (pred_img - gt_img) * mask

def calculate_batch_nmse(imgs1, imgs2, masks,channel=0):
    """Calculate NMSE for the masked areas between two batches of images."""
    epsilon = 1e-8
    numerator = 0.0
    denominator = 0.0
    nmse_avg = 0

    for img1, img2, mask in zip(imgs1, imgs2, masks):
        # Extract the specified channel
        img1_channel = img1[channel]
        img2_channel = img2[channel]
        mask_channel = mask.squeeze()  # Assuming mask has shape [1,H,W]
        
        # Apply mask
        masked_gt_img = img1_channel * mask_channel
        masked_est_img = img2_channel * mask_channel

        numerator += np.sum((masked_est_img - masked_gt_img) ** 2)
        denominator += np.sum(masked_gt_img ** 2)

        nmse_avg += (numerator + epsilon) / (denominator + epsilon)
    return nmse_avg/len(imgs1)

def calculate_psnr_normalized(img1, img2, mask=None):
    """Calculate PSNR for normalized images with explicit data range of 1.0"""
    if mask is not None:
        img1 = img1 * mask
        img2 = img2 * mask
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    # Explicitly set data_range=1.0 for normalized images
    return 20 * np.log10(1.0 / np.sqrt(mse))
def calculate_batch_psnr(imgs1, imgs2, masks, channel=0):

    psnr_sum = 0.0
    for img1, img2, mask in zip(imgs1, imgs2, masks):
        # Extract the specified channel
        img1_channel = img1[channel]
        img2_channel = img2[channel]
        mask_channel = mask.squeeze()  # Assuming mask has shape [1,H,W]
        
        # Apply mask
        masked_img1 = img1_channel * mask_channel
        masked_img2 = img2_channel * mask_channel
        
        # Calculate PSNR
        psnr_sum += psnr_sk(masked_img2, masked_img1, data_range=1)
    
    return psnr_sum / len(imgs1)

def calculate_batch_ssim(imgs1, imgs2, masks, channel=0):
    """Calculate SSIM for the masked areas between two batches of images."""
    ssim_sum = 0.0
    for img1, img2, mask in zip(imgs1, imgs2, masks):

        img1_channel = img1[channel]
        img2_channel = img2[channel]
        mask_channel = mask.squeeze()  # Assuming mask has shape [1,H,W]
        
        # Apply mask
        masked_img1 = img1_channel * mask_channel
        masked_img2 = img2_channel * mask_channel
        ssim_sum += ssim_sk(masked_img2, masked_img1, data_range=1)
    return ssim_sum / len(imgs1)

def create_water_baseline(mask, num_channels):

    window_min = 1000 - 4000/2  # -1000
    window_max = 1000 + 4000/2  # 3000
    normalized_0_HU = (0 - window_min) / (window_max - window_min)  # = 0.25
    
    # Create the water baseline: 0 HU within mask, implicitly -1000 HU (normalized to 0) outside
    water_baseline = torch.ones_like(mask) * normalized_0_HU * mask
    
    # Repeat for all channels to match CT/CBCT tensor shape
    if num_channels > 1:
        water_baseline = water_baseline.repeat(1, num_channels, 1, 1)
    
    return water_baseline

    import numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (binary_closing, binary_erosion, remove_small_holes,
                                remove_small_objects, disk)
from skimage.measure import label, regionprops, find_contours

def compute_batch_crop_size(mask_tensor, min_boundary=5):
    import numpy as np
    
    # Convert to numpy if it's a torch tensor
    if hasattr(mask_tensor, 'cpu'):
        mask_batch = mask_tensor.cpu().numpy()
    else:
        mask_batch = mask_tensor
    
    batch_size, channel, height, width = mask_batch.shape
    
    # Initialize with extreme values
    r0_min = height  # Start with max possible row
    r1_max = -1      # Start with min possible row
    c0_min = width   # Start with max possible col
    c1_max = -1      # Start with min possible col
    
    # Process each mask in the batch
    for i in range(batch_size):
        mask = mask_batch[i].squeeze()
        
        # Find non-zero pixels in this mask
        nonzero_coords = np.nonzero(mask)
        
        # Skip if mask is empty
        if len(nonzero_coords[0]) == 0:
            continue
            
        h_coords, w_coords = nonzero_coords
        
        # Get bounding box for this mask
        r0 = h_coords.min()
        r1 = h_coords.max()
        c0 = w_coords.min()
        c1 = w_coords.max()
        
        # Update global bounding box
        r0_min = min(r0_min, r0)
        r1_max = max(r1_max, r1)
        c0_min = min(c0_min, c0)
        c1_max = max(c1_max, c1)
    
    # Handle case where all masks are empty
    if r1_max == -1:
        return 0, height-1, 0, width-1
    
    # Add minimum boundary padding
    r0_min = max(0, r0_min - min_boundary)
    r1_max = min(height - 1, r1_max + min_boundary)
    c0_min = max(0, c0_min - min_boundary)
    c1_max = min(width - 1, c1_max + min_boundary)
    
    # Calculate current dimensions
    current_height = r1_max - r0_min + 1
    current_width = c1_max - c0_min + 1
    
    
    return r0_min, r1_max, c0_min, c1_max


def zoom_cbct_roi(img, center=(0.5, 0.5), width_height_ratio=1.0, crop_width_ratio=0.2):

    H, W = img.shape
    
    # Convert normalized center to pixel coordinates
    cy, cx = center
    cy = np.clip(cy, 0.0, 1.0)
    cx = np.clip(cx, 0.0, 1.0)
    cy_px = int(round(cy * (H - 1)))
    cx_px = int(round(cx * (W - 1)))
    
    # Calculate crop dimensions
    crop_width = int(round(crop_width_ratio * W))
    crop_height = int(round(crop_width / width_height_ratio))
    
    # Ensure odd dimensions for centered cropping
    if crop_width % 2 == 0:
        crop_width = max(1, crop_width - 1)
    if crop_height % 2 == 0:
        crop_height = max(1, crop_height - 1)
    
    # Calculate crop boundaries
    y0 = cy_px - crop_height // 2
    x0 = cx_px - crop_width // 2
    y1 = y0 + crop_height
    x1 = x0 + crop_width
    
    # Clamp to image boundaries and shift if necessary
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y1 > H:
        y0 -= (y1 - H)
        y1 = H
    if x1 > W:
        x0 -= (x1 - W)
        x1 = W
    
    # Final safety clamp
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(H, y1)
    x1 = min(W, x1)
    
    # Extract the crop
    cropped_img = img[y0:y1, x0:x1]
    
    return cropped_img


from matplotlib.patches import Rectangle

def add_zoom_overlay(ax, img, *,
                     roi_center=(0.5, 0.7),   # (x_frac, y_frac) in [0,1]
                     roi_w=0.4, roi_h=None,  # ROI size as fractions; default square
                     inset_w_frac=0.35,      # inset width as fraction of parent axes
                     pad_frac=0.02,          # inset padding from top-right corner
                     cmap=None, norm=None):

    if roi_h is None:
        roi_h = roi_w

    # --- compute ROI in DATA coordinates (pixels) ---
    H, W = img.shape[-2], img.shape[-1]  # accept (H,W) or (C,H,W)
    cx, cy = roi_center
    x0 = max(0, (cx - roi_w/2) * (W - 1))
    x1 = min(W - 1, (cx + roi_w/2) * (W - 1))
    y0 = max(0, (cy - roi_h/2) * (H - 1))
    y1 = min(H - 1, (cy + roi_h/2) * (H - 1))

    # --- draw ROI rect in *data coords* (exactly matches pixels) ---
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                     fill=False, edgecolor='red', linewidth=2.0, linestyle=':')
    ax.add_patch(rect)

    # --- size inset to match ROI aspect (no distortion) ---
    roi_aspect = (x1 - x0) / max(1e-9, (y1 - y0))     # width / height
    inset_h_frac = inset_w_frac / roi_aspect          # H = W / aspect

    # --- overlay inset in parent-axes coordinates (top-right) ---
    left   = 1.0 - inset_w_frac - pad_frac
    bottom = 1.0 - inset_h_frac - pad_frac
    axins  = ax.inset_axes([left, bottom, inset_w_frac, inset_h_frac],
                           transform=ax.transAxes)
    # show full image, then crop view → preserves geometry
    axins.imshow(img, cmap=cmap, norm=norm, origin='upper', aspect='equal')
    axins.set_xlim(x0, x1)
    axins.set_ylim(y1, y0)  # invert y for origin='upper'
    axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_linewidth(0.8)
    return axins

def displaywindow(img,
                  display_window_center,
                  display_window_width,
                  base_window_min=-1000.0,
                  base_window_max=3000.0,
                  input_range=(0.0, 1.0)):

    a, b = input_range
    if display_window_width <= 0:
        raise ValueError("display_window_width must be > 0")

    # 1) De-normalize to HU (limited to the base window used during preprocessing)
    scale = (base_window_max - base_window_min) / (b - a)
    img_hu = (img - a) * scale + base_window_min

    # 2) Apply display window
    wmin = display_window_center - display_window_width / 2.0
    wmax = display_window_center + display_window_width / 2.0
    img_w = np.clip(img_hu, wmin, wmax)

    # 3) Normalize to [0, 1] using the display window bounds
    out = (img_w - wmin) / (wmax - wmin)
    return out

