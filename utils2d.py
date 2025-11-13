import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def dice_coefficient_2d(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient for 2D images
    Args:
        pred: predicted segmentation
        target: ground truth segmentation
        smooth: smoothing factor
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def jacobian_determinant_2d(disp):
    """
    Calculate the Jacobian determinant of a 2D displacement field
    Args:
        disp: displacement field of shape (batch, 2, height, width)
    Returns:
        Jacobian determinant at each pixel
    """
    # Get displacement gradients
    _, _, H, W = disp.shape
    
    # Calculate gradients using finite differences
    # du/dx, du/dy
    du_dx = disp[:, 0, :, 1:] - disp[:, 0, :, :-1]  # Shape: (B, H, W-1)
    du_dy = disp[:, 0, 1:, :] - disp[:, 0, :-1, :]  # Shape: (B, H-1, W)
    
    # dv/dx, dv/dy
    dv_dx = disp[:, 1, :, 1:] - disp[:, 1, :, :-1]  # Shape: (B, H, W-1)
    dv_dy = disp[:, 1, 1:, :] - disp[:, 1, :-1, :]  # Shape: (B, H-1, W)
    
    # Crop to same size
    du_dx = du_dx[:, :-1, :]  # (B, H-1, W-1)
    du_dy = du_dy[:, :, :-1]  # (B, H-1, W-1)
    dv_dx = dv_dx[:, :-1, :]  # (B, H-1, W-1)
    dv_dy = dv_dy[:, :, :-1]  # (B, H-1, W-1)
    
    # Jacobian matrix:
    # J = [[1 + du/dx, du/dy],
    #      [dv/dx, 1 + dv/dy]]
    # det(J) = (1 + du/dx)(1 + dv/dy) - (du/dy)(dv/dx)
    det = (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx
    
    return det


def calculate_tre_2d(pred_landmarks, true_landmarks):
    """
    Calculate Target Registration Error (TRE) for 2D landmarks
    Args:
        pred_landmarks: predicted landmark positions (N, 2)
        true_landmarks: ground truth landmark positions (N, 2)
    Returns:
        TRE values for each landmark
    """
    if isinstance(pred_landmarks, torch.Tensor):
        pred_landmarks = pred_landmarks.cpu().numpy()
    if isinstance(true_landmarks, torch.Tensor):
        true_landmarks = true_landmarks.cpu().numpy()
    
    distances = np.sqrt(np.sum((pred_landmarks - true_landmarks) ** 2, axis=1))
    return distances


def warp_landmarks_2d(landmarks, flow):
    """
    Warp landmarks using displacement field
    Args:
        landmarks: landmark coordinates (N, 2) in [x, y] format
        flow: displacement field (1, 2, H, W)
    Returns:
        warped landmarks (N, 2)
    """
    if isinstance(landmarks, np.ndarray):
        landmarks = torch.from_numpy(landmarks).float()
    
    # Normalize coordinates to [-1, 1] for grid_sample
    H, W = flow.shape[2:]
    landmarks_norm = landmarks.clone()
    landmarks_norm[:, 0] = 2 * landmarks[:, 0] / (W - 1) - 1  # x coordinate
    landmarks_norm[:, 1] = 2 * landmarks[:, 1] / (H - 1) - 1  # y coordinate
    
    # Reshape for grid_sample: (1, N, 1, 2)
    landmarks_grid = landmarks_norm.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
    
    # Sample displacement at landmark locations
    # flow shape: (1, 2, H, W) -> need (1, H, W, 2) for grid_sample
    flow_permuted = flow.permute(0, 2, 3, 1)  # (1, H, W, 2)
    
    # Sample flow at landmark positions
    sampled_flow = F.grid_sample(flow, landmarks_grid, align_corners=True, mode='bilinear')
    sampled_flow = sampled_flow.squeeze(0).squeeze(2).t()  # (N, 2)
    
    # Add displacement to original coordinates
    warped_landmarks = landmarks + sampled_flow
    
    return warped_landmarks


def visualize_flow_field_2d(flow, step=10, scale=1.0, save_path=None):
    """
    Visualize 2D flow field as arrows
    Args:
        flow: displacement field (1, 2, H, W) or (2, H, W)
        step: step size for arrow sampling
        scale: scale factor for arrow length
        save_path: path to save the visualization
    """
    if len(flow.shape) == 4:
        flow = flow[0]  # Remove batch dimension
    
    flow_np = flow.cpu().numpy() if isinstance(flow, torch.Tensor) else flow
    
    H, W = flow_np.shape[1:]
    
    # Create coordinate grids
    y, x = np.mgrid[0:H:step, 0:W:step]
    
    # Sample flow at grid points
    u = flow_np[0, ::step, ::step] * scale
    v = flow_np[1, ::step, ::step] * scale
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7)
    plt.xlim(0, W)
    plt.ylim(H, 0)  # Flip y-axis to match image coordinates
    plt.title('Displacement Field Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_checkerboard_2d(img1, img2, block_size=20):
    """
    Create checkerboard visualization of two images
    Args:
        img1, img2: input images (H, W) or (1, H, W)
        block_size: size of checkerboard blocks
    Returns:
        checkerboard image
    """
    if len(img1.shape) == 3:
        img1 = img1[0]
    if len(img2.shape) == 3:
        img2 = img2[0]
    
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    H, W = img1.shape
    checkerboard = np.zeros_like(img1)
    
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                checkerboard[i:i+block_size, j:j+block_size] = img1[i:i+block_size, j:j+block_size]
            else:
                checkerboard[i:i+block_size, j:j+block_size] = img2[i:i+block_size, j:j+block_size]
    
    return checkerboard


def save_registration_results_2d(moving, fixed, warped, flow, save_dir, filename_prefix):
    """
    Save comprehensive registration results
    Args:
        moving, fixed, warped: images (1, 1, H, W)
        flow: displacement field (1, 2, H, W)
        save_dir: directory to save results
        filename_prefix: prefix for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy
    moving_np = moving[0, 0].cpu().numpy()
    fixed_np = fixed[0, 0].cpu().numpy()
    warped_np = warped[0, 0].cpu().numpy()
    
    # Save individual images
    Image.fromarray((moving_np * 255).astype(np.uint8)).save(
        os.path.join(save_dir, f'{filename_prefix}_moving.png'))
    Image.fromarray((fixed_np * 255).astype(np.uint8)).save(
        os.path.join(save_dir, f'{filename_prefix}_fixed.png'))
    Image.fromarray((warped_np * 255).astype(np.uint8)).save(
        os.path.join(save_dir, f'{filename_prefix}_warped.png'))
    
    # Create and save checkerboard
    checkerboard = create_checkerboard_2d(warped_np, fixed_np)
    Image.fromarray((checkerboard * 255).astype(np.uint8)).save(
        os.path.join(save_dir, f'{filename_prefix}_checkerboard.png'))
    
    # Save flow visualization
    flow_vis_path = os.path.join(save_dir, f'{filename_prefix}_flow.png')
    visualize_flow_field_2d(flow, save_path=flow_vis_path)
    
    # Save difference images
    diff_before = np.abs(moving_np - fixed_np)
    diff_after = np.abs(warped_np - fixed_np)
    
    Image.fromarray((diff_before * 255).astype(np.uint8)).save(
        os.path.join(save_dir, f'{filename_prefix}_diff_before.png'))
    Image.fromarray((diff_after * 255).astype(np.uint8)).save(
        os.path.join(save_dir, f'{filename_prefix}_diff_after.png'))


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_size=(1, 1, 256, 256)):
    """
    Print model summary including parameter count and output shapes
    """
    print("Model Summary:")
    print("=" * 50)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Test forward pass to get output shapes
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        try:
            output = model(dummy_input, dummy_input)
            if isinstance(output, tuple):
                for i, out in enumerate(output):
                    print(f"Output {i+1} shape: {out.shape}")
            else:
                print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Could not determine output shape: {e}")
    
    print("=" * 50)


def setup_directories(base_dir='.'):
    """
    Setup standard directory structure for VoxelMorph 2D
    """
    dirs = [
        'data/train/moving',
        'data/train/fixed', 
        'data/test/moving',
        'data/test/fixed',
        'models',
        'logs',
        'results',
        'results/moved'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")
    
    print("\nDirectory structure created successfully!")
    print("Please add your training and test images to the appropriate directories.")


if __name__ == '__main__':
    # Test utility functions
    print("Testing VoxelMorph 2D utilities...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}, Std: {meter.std}")
    
    # Test Jacobian determinant calculation
    dummy_flow = torch.randn(1, 2, 64, 64)
    jac_det = jacobian_determinant_2d(dummy_flow)
    print(f"Jacobian determinant shape: {jac_det.shape}")
    print(f"Mean Jacobian determinant: {jac_det.mean().item():.4f}")
    
    print("All tests passed!")