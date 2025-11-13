'''
VoxelMorph-2D: 2D version of VoxelMorph Network
Adapted from the original 3D VoxelMorph model for 2D image registration
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class SpatialTransformer2D(nn.Module):
    """
    2D Spatial Transformer
    """
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer2D, self).__init__()
        self.mode = mode

        # 创建采样网格
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # (2, H, W)
        grid = torch.unsqueeze(grid, 0)  # (1, 2, H, W)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # 计算新的位置
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # 将网格坐标归一化到[-1, 1]范围
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # 调整维度顺序为(B, H, W, 2)，并交换x和y坐标
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)

class Conv2DBlock(nn.Module):
    """
    2D Convolutional Layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.conv(x))

class UNet2D(nn.Module):

    def __init__(self, in_channel=2, full_size=True):
        super(UNet2D, self).__init__()
        self.full_size = full_size

        #编码器部分
        self.enc1 = Conv2DBlock(2, 16, kernel_size=4, stride=2, padding=1)
        self.enc2 = Conv2DBlock(16, 32, kernel_size=4, stride=2, padding=1)
        self.enc3 = Conv2DBlock(32, 32, kernel_size=4, stride=2, padding=1)
        self.enc4 = Conv2DBlock(32, 32, kernel_size=4, stride=2, padding=1)

        #解码器部分
        self.dec5 = Conv2DBlock(32, 32)
        self.dec4 = Conv2DBlock(32+32, 32)
        self.dec3 = Conv2DBlock(32+32, 32)
        self.dec2 = Conv2DBlock(32+16, 32)
        self.dec1 = Conv2DBlock(32, 32)

        if self.full_size:
            self.dec0 = Conv2DBlock(32+2, 16)
       
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.flow = nn.Conv2d(16, 2, kernel_size=3, padding=1)
         # 初始化权重和偏置为小值，用均值为0，标准差1e-5的正态分布初始化权重，避免训练初期产生较大的流场
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, fixed, moving):
        # 拼接固定图像和浮动图像（保留原始全分辨率输入用于最终拼接）
        x0 = torch.cat([fixed, moving], dim=1)

        # 特征提取
        x1 = self.enc1(x0)  # 1/2
        x2 = self.enc2(x1) #1/4
        x3 = self.enc3(x2) #1/8
        x4 = self.enc4(x3) #1/16

        # 解码阶段 - 三次卷积 + 上采样 + 拼接
        x = self.upsample(self.dec5(x4))
        x = torch.cat([x, x3], dim=1)
        x = self.upsample(self.dec4(x))
        x = torch.cat([x, x2], dim=1)
        x = self.upsample(self.dec3(x))
        x = torch.cat([x, x1], dim=1)
        # 在1/2分辨率进行两次解码卷积
        x = self.dec2(x)
        x = self.dec1(x)
        # 上采样到全分辨率，与原始输入（2通道）拼接并卷积
        if self.full_size:
            x = self.upsample(x)
            x = torch.cat([x, x0], dim=1)
            x = self.dec0(x)

        flow = self.flow(x)

        return flow

class VoxelMorph2D(nn.Module):
    """
    完整的VoxelMorph 2D模型，包含U-Net和空间变换器
    """
    def __init__(self, img_size=(256, 256), full_size=True):
        super(VoxelMorph2D, self).__init__()
        
        # U-Net网络
        self.unet = UNet2D(full_size=full_size)
        
        # 空间变换器
        self.spatial_transformer = SpatialTransformer2D(img_size)

    def forward(self, moving, fixed):
        # 通过U-Net生成流场
        flow = self.unet(moving, fixed)
        
        # 使用空间变换器对移动图像进行变形
        warped = self.spatial_transformer(moving, flow)
        
        return warped, flow

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试模型
    model = VoxelMorph2D(img_size=(256, 256)).to(device)
    
    moving = torch.randn(1, 1, 256, 256).to(device)
    fixed = torch.randn(1, 1, 256, 256).to(device)
    
    with torch.no_grad():
        warped, flow = model(moving, fixed)
    
    print(f"Moving shape: {moving.shape}")
    print(f"Fixed shape: {fixed.shape}")
    print(f"Warped shape: {warped.shape}")
    print(f"Flow shape: {flow.shape}")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")