import os, glob
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageDataset2D(Dataset):
    def __init__(self, moving_dir, fixed_dir, transforms=None):
        """
        Dataset for 2D image registration training
        
        Args:
            moving_dir: Directory containing moving images
            fixed_dir: Directory containing fixed images
            transforms: Transforms to apply to the images
        """
        self.moving_paths = sorted(glob.glob(os.path.join(moving_dir, '*.png')) + 
                                  glob.glob(os.path.join(moving_dir, '*.jpg')))
        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_dir, '*.png')) + 
                                 glob.glob(os.path.join(fixed_dir, '*.jpg')))
        
        assert len(self.moving_paths) > 0, f"No images found in {moving_dir}"
        assert len(self.fixed_paths) > 0, f"No images found in {fixed_dir}"
        
        self.transforms = transforms

    def __getitem__(self, index):
        # Get paths for moving and fixed images
        moving_path = self.moving_paths[index % len(self.moving_paths)]
        fixed_path = self.fixed_paths[index % len(self.fixed_paths)]
        
        # Load images
        moving_img = Image.open(moving_path).convert('L')  # Convert to grayscale
        fixed_img = Image.open(fixed_path).convert('L')    # Convert to grayscale
        
        # Convert to numpy arrays
        moving_img = np.array(moving_img) / 255.0  # Normalize to [0, 1]
        fixed_img = np.array(fixed_img) / 255.0    # Normalize to [0, 1]
        
        # Add channel dimension
        moving_img = moving_img[None, ...]
        fixed_img = fixed_img[None, ...]
        
        # Apply transforms if available
        if self.transforms:
            moving_img, fixed_img = self.transforms([moving_img, fixed_img])
        
        # Convert to torch tensors
        moving_img = torch.from_numpy(moving_img.astype(np.float32))
        fixed_img = torch.from_numpy(fixed_img.astype(np.float32))
        
        return moving_img, fixed_img

    def __len__(self):
        return max(len(self.moving_paths), len(self.fixed_paths))


class ImageInferDataset2D(Dataset):
    def __init__(self, moving_dir, fixed_dir, moving_seg_dir=None, fixed_seg_dir=None, transforms=None):
        """
        Dataset for 2D image registration inference with optional segmentation masks
        
        Args:
            moving_dir: Directory containing moving images
            fixed_dir: Directory containing fixed images
            moving_seg_dir: Directory containing moving segmentation masks (optional)
            fixed_seg_dir: Directory containing fixed segmentation masks (optional)
            transforms: Transforms to apply to the images
        """
        self.moving_paths = sorted(glob.glob(os.path.join(moving_dir, '*.png')) + 
                                  glob.glob(os.path.join(moving_dir, '*.jpg')))
        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_dir, '*.png')) + 
                                 glob.glob(os.path.join(fixed_dir, '*.jpg')))
        
        self.has_segmentations = False
        if moving_seg_dir and fixed_seg_dir:
            self.moving_seg_paths = sorted(glob.glob(os.path.join(moving_seg_dir, '*.png')) + 
                                          glob.glob(os.path.join(moving_seg_dir, '*.jpg')))
            self.fixed_seg_paths = sorted(glob.glob(os.path.join(fixed_seg_dir, '*.png')) + 
                                         glob.glob(os.path.join(fixed_seg_dir, '*.jpg')))
            
            assert len(self.moving_seg_paths) > 0, f"No segmentation masks found in {moving_seg_dir}"
            assert len(self.fixed_seg_paths) > 0, f"No segmentation masks found in {fixed_seg_dir}"
            self.has_segmentations = True
        
        assert len(self.moving_paths) > 0, f"No images found in {moving_dir}"
        assert len(self.fixed_paths) > 0, f"No images found in {fixed_dir}"
        
        self.transforms = transforms

    def __getitem__(self, index):
        # Get paths for moving and fixed images
        moving_path = self.moving_paths[index % len(self.moving_paths)]
        fixed_path = self.fixed_paths[index % len(self.fixed_paths)]
        
        # Load images
        moving_img = Image.open(moving_path).convert('L')  # Convert to grayscale
        fixed_img = Image.open(fixed_path).convert('L')    # Convert to grayscale
        
        # Convert to numpy arrays
        moving_img = np.array(moving_img) / 255.0  # Normalize to [0, 1]
        fixed_img = np.array(fixed_img) / 255.0    # Normalize to [0, 1]
        
        # Add channel dimension
        moving_img = moving_img[None, ...]
        fixed_img = fixed_img[None, ...]
        
        if self.has_segmentations:
            # Get paths for segmentation masks
            moving_seg_path = self.moving_seg_paths[index % len(self.moving_seg_paths)]
            fixed_seg_path = self.fixed_seg_paths[index % len(self.fixed_seg_paths)]
            
            # Load segmentation masks
            moving_seg = Image.open(moving_seg_path).convert('L')
            fixed_seg = Image.open(fixed_seg_path).convert('L')
            
            # Convert to numpy arrays
            moving_seg = np.array(moving_seg)
            fixed_seg = np.array(fixed_seg)
            
            # Add channel dimension
            moving_seg = moving_seg[None, ...]
            fixed_seg = fixed_seg[None, ...]
            
            # Apply transforms if available
            if self.transforms:
                moving_img, moving_seg = self.transforms([moving_img, moving_seg])
                fixed_img, fixed_seg = self.transforms([fixed_img, fixed_seg])
            
            # Convert to torch tensors
            moving_img = torch.from_numpy(moving_img.astype(np.float32))
            fixed_img = torch.from_numpy(fixed_img.astype(np.float32))
            moving_seg = torch.from_numpy(moving_seg.astype(np.int16))
            fixed_seg = torch.from_numpy(fixed_seg.astype(np.int16))
            
            return moving_img, fixed_img, moving_seg, fixed_seg
        else:
            # Apply transforms if available
            if self.transforms:
                moving_img, fixed_img = self.transforms([moving_img, fixed_img])
            
            # Convert to torch tensors
            moving_img = torch.from_numpy(moving_img.astype(np.float32))
            fixed_img = torch.from_numpy(fixed_img.astype(np.float32))
            
            return moving_img, fixed_img

    def __len__(self):
        return max(len(self.moving_paths), len(self.fixed_paths))


# 在ImageInferDataset2D类之前添加以下代码

class ImageDataset2DCrossPairs(Dataset):
    def __init__(self, moving_dir, fixed_dir, transforms=None):
        """
        Dataset for 2D image registration training with all possible cross-directory image pairs
        
        Args:
            moving_dir: Directory containing moving images
            fixed_dir: Directory containing fixed images
            transforms: Transforms to apply to the images
        """
        self.moving_paths = sorted(glob.glob(os.path.join(moving_dir, '*.png')) + 
                                  glob.glob(os.path.join(moving_dir, '*.jpg')))
        self.fixed_paths = sorted(glob.glob(os.path.join(fixed_dir, '*.png')) + 
                                 glob.glob(os.path.join(fixed_dir, '*.jpg')))
        
        assert len(self.moving_paths) > 0, f"No images found in {moving_dir}"
        assert len(self.fixed_paths) > 0, f"No images found in {fixed_dir}"
        
        self.transforms = transforms
        print(f"创建了包含 {len(self.moving_paths)} 张移动图像和 {len(self.fixed_paths)} 张固定图像的数据集，共有 {len(self)} 对可能的组合")

    def __getitem__(self, index):
        # 计算图像对的索引
        moving_idx = index // len(self.fixed_paths)
        fixed_idx = index % len(self.fixed_paths)
        
        # 获取图像路径
        moving_path = self.moving_paths[moving_idx]
        fixed_path = self.fixed_paths[fixed_idx]
        
        # 加载图像
        moving_img = Image.open(moving_path).convert('L')  # 转换为灰度图
        fixed_img = Image.open(fixed_path).convert('L')    # 转换为灰度图
        
        # 转换为numpy数组
        moving_img = np.array(moving_img) / 255.0  # 归一化到[0, 1]
        fixed_img = np.array(fixed_img) / 255.0    # 归一化到[0, 1]
        
        # 添加通道维度
        moving_img = moving_img[None, ...]
        fixed_img = fixed_img[None, ...]
        
        # 应用变换（如果有）
        if self.transforms:
            moving_img, fixed_img = self.transforms([moving_img, fixed_img])
        
        # 转换为torch张量
        moving_img = torch.from_numpy(moving_img.astype(np.float32))
        fixed_img = torch.from_numpy(fixed_img.astype(np.float32))
        
        return moving_img, fixed_img

    def __len__(self):
        # 所有可能的跨目录图像对组合数量
        return len(self.moving_paths) * len(self.fixed_paths)