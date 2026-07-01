import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Alias used at inference time (same pipeline as validation)
infer_transform = val_transform


class MRIDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        arr = np.array(img)
        if arr.max() == 0:
            arr = np.full_like(arr, 128)
        arr = _clahe.apply(arr)
        img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


def get_image_paths_and_labels(dataset_path):
    image_paths, labels = [], []
    for label, class_name in enumerate(["Normal", "Sick"]):
        class_dir = os.path.join(dataset_path, class_name)
        for patient_name in sorted(os.listdir(class_dir)):
            patient_path = os.path.join(class_dir, patient_name)
            if not os.path.isdir(patient_path):
                continue
            for series_name in sorted(os.listdir(patient_path)):
                series_path = os.path.join(patient_path, series_name)
                if not os.path.isdir(series_path):
                    continue
                for f in sorted(os.listdir(series_path)):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        image_paths.append(os.path.join(series_path, f))
                        labels.append(label)
    return image_paths, labels


def get_patients(dataset_path):
    """Return (patient_dirs, labels) — one entry per patient directory."""
    patient_dirs, labels = [], []
    for label, class_name in enumerate(["Normal", "Sick"]):
        class_dir = os.path.join(dataset_path, class_name)
        for name in sorted(os.listdir(class_dir)):
            path = os.path.join(class_dir, name)
            if os.path.isdir(path):
                patient_dirs.append(path)
                labels.append(label)
    return patient_dirs, labels


def patient_to_slices(patient_dirs, labels):
    """Expand a list of patient directories into individual slice paths + labels.
    Goes exactly one subdir deep (patient/series/image) to avoid counting
    duplicate nested dirs that exist in Normal patients."""


    
    paths, slice_labels = [], []
    for patient_dir, label in zip(patient_dirs, labels):
        for series_name in sorted(os.listdir(patient_dir)):
            series_path = os.path.join(patient_dir, series_name)
            if not os.path.isdir(series_path):
                continue
            for f in sorted(os.listdir(series_path)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    paths.append(os.path.join(series_path, f))
                    slice_labels.append(label)
    return paths, slice_labels
