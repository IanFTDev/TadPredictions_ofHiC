import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class SegmentationDataset(Dataset):
    def __init__(self, dataset_root, file_path, num_classes=2, target_size=(512, 512), 
                 mode='train', separator=' ', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.dataset_root = dataset_root
        self.num_classes = num_classes
        self.target_size = target_size
        self.mode = mode
        self.separator = separator
        
        # Normalization
        self.normalize = transforms.Normalize(mean=mean, std=std)
        
        # Load file paths
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(separator)
                    if len(parts) >= 2:
                        img_path = os.path.join(dataset_root, parts[0])
                        mask_path = os.path.join(dataset_root, parts[1])
                        self.samples.append((img_path, mask_path))
        
        print(f"Loaded {len(self.samples)} samples for {mode} mode")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to PIL for transforms
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        # Apply transforms
        image, mask = self.apply_transforms(image, mask)
        
        # Convert to tensor
        image = F.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Normalize image
        image = self.normalize(image)
        
        return image, mask
    
    def apply_transforms(self, image, mask):
        # Resize
        # image = F.resize(image, self.target_size, interpolation=Image.BILINEAR)
        # mask = F.resize(mask, self.target_size, interpolation=Image.NEAREST)
        
        # if self.mode == 'train':
        #     # Random horizontal  adn veritical flip
        #     if torch.rand(1) < 0.5:
        #         image = F.hflip(image)
        #         mask = F.hflip(mask)
        #         image = F.vflip(image)
        #         mask = F.vflip(mask)
            
            
        
        return image, mask