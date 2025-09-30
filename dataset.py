import os
import glob
import SimpleITK as sitk
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Normalizing:
    def __call__(self, tensor):
        tensor = tensor.float()
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        if tensor_max > tensor_min:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        else:
            print("Normalization error: constant intensity in image.")
        return tensor

class Windowing:
    def __init__(self, window_center, window_width):
        self.window_center = window_center
        self.window_width = window_width

    def __call__(self, image):
        window_min = self.window_center - self.window_width / 2
        window_max = self.window_center + self.window_width / 2
        image = torch.clip(image, window_min, window_max)
        image = (image - window_min) / (window_max - window_min)  # normalize
        return image

class CBCTDataset(Dataset):
    def __init__(self, patient_dirs, mode="train", image_size=64, dataset_type="pelvis"):
        self.patient_dirs = patient_dirs
        self.image_size = image_size
        self.dataset_type = dataset_type.lower()
        
        self.pct_slices = []
        self.cbct_slices = []
        self.mask_slices = []

        for patient_dir in self.patient_dirs:
            pct_path = os.path.join(patient_dir, 'aligned_pct.mha')
            cbct_path = os.path.join(patient_dir, 'cbct.mha')
            mask_path = os.path.join(patient_dir, 'mask.mha')

            pct_image = sitk.ReadImage(pct_path)
            cbct_image = sitk.ReadImage(cbct_path)
            mask_image = sitk.ReadImage(mask_path)
            
            pct_array = sitk.GetArrayFromImage(pct_image).astype(np.float32)
            cbct_array = sitk.GetArrayFromImage(cbct_image).astype(np.float32)
            mask_array = sitk.GetArrayFromImage(mask_image).astype(np.float32)
            
            # Add each slice to their respective lists
            self.pct_slices.extend(pct_array)
            self.cbct_slices.extend(cbct_array)
            self.mask_slices.extend(mask_array)

        # Define windowing parameters based on dataset type
        if self.dataset_type == "pelvis":
            window_params = [
                {'center': 1000, 'width': 4000},   # Bone window
                {'center': 50, 'width': 400},      # Soft tissue window
                {'center': 600, 'width': 3000},    # Intermediate window
            ]
        elif self.dataset_type == "brain":
            window_params = [
                {'center': 1000, 'width': 4000},   # Bone window
                {'center': 50, 'width': 400},      # Soft tissue/brain window
                {'center': 35, 'width': 80},       # Brain-specific window
            ]
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}. Choose 'pelvis' or 'brain'.")

        # Define transformations for PCT
        self.pct_transforms = []
        for params in window_params:
            self.pct_transforms.append(transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([self.image_size, self.image_size]),
                Windowing(window_center=params['center'], window_width=params['width']),
            ]))

        # Define transformations for CBCT
        self.cbct_transforms = []
        for params in window_params:
            self.cbct_transforms.append(transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([self.image_size, self.image_size]),
                Windowing(window_center=params['center'], window_width=params['width']),
            ]))

        # Define mask transformation
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([self.image_size, self.image_size]),
            Normalizing(),  # Masks are typically binary, so we can normalize
        ])

    def __getitem__(self, index):
        # Ensure index wraps around if the number of slices differ
        pct_slice = self.pct_slices[index % len(self.pct_slices)]
        cbct_slice = self.cbct_slices[index % len(self.cbct_slices)]
        mask_slice = self.mask_slices[index % len(self.mask_slices)]
        
        # Apply transformations to PCT slices
        pct_images = []
        for transform in self.pct_transforms:
            img = transform(pct_slice)  # shape (1, H, W)
            img = img.squeeze(0)        # shape (H, W)
            pct_images.append(img)
        pct_image_tensor = torch.stack(pct_images, dim=0)  # shape (3, H, W)

        # Apply transformations to CBCT slices
        cbct_images = []
        for transform in self.cbct_transforms:
            img = transform(cbct_slice)
            img = img.squeeze(0)
            cbct_images.append(img)
        cbct_image_tensor = torch.stack(cbct_images, dim=0)  # shape (3, H, W)

        # Transform mask
        mask_slice = self.mask_transform(mask_slice)
        return {
            "pct": pct_image_tensor,
            "cbct": cbct_image_tensor,
            "mask": mask_slice
        }


    def __len__(self):
        return max(len(self.pct_slices), len(self.cbct_slices), len(self.mask_slices))

def create_paired_datasets(data_dir, split_ratio=0.8, image_size=64, dataset_type="pelvis"):
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
    if len(patient_dirs) == 0:
        raise ValueError(f"No patient directories found in {data_dir}. Please check the path and structure.")
    
    # Split the patient directories into training and testing sets
    train_dirs, test_dirs = train_test_split(patient_dirs, train_size=split_ratio, random_state=42)
    print(f"Dataset type: {dataset_type}")
    print("Train directories:", train_dirs)
    print("Test directories:", test_dirs)
    print("Length of train directories:", len(train_dirs))
    print("Length of test directories:", len(test_dirs))
    
    # Create datasets with specified dataset type
    train_set = CBCTDataset(patient_dirs=train_dirs, mode="train", 
                           image_size=image_size, dataset_type=dataset_type)
    test_dataset = CBCTDataset(patient_dirs=test_dirs, mode="test", 
                              image_size=image_size, dataset_type=dataset_type)
    
    return train_set, test_dataset
