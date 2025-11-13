import glob
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from natsort import natsorted
import random
from PIL import Image

# 导入自定义模块
from vxm2d import VoxelMorph2D
import utils2d
import sys
sys.path.append(r'E:\SmileCode\data')
import datasets2d


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

same_seeds(24)


def main():
    # 数据目录
    test_moving_dir = r'E:\RDP\data\test\moving'  # 使用训练数据作为测试
    test_fixed_dir = r'E:\RDP\data\test\fixed'    # 使用训练数据作为测试
    
    # 结果保存目录
    result_dir = 'VoxelMorph/results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 形变过后的图像保存目录
    moved_dir = 'VoxelMorph/results/moved'
    if not os.path.exists(moved_dir):
        os.makedirs(moved_dir)
    
    # 初始化数据集
    test_set = datasets2d.ImageInferDataset2D(test_moving_dir, test_fixed_dir)
    
    # 数据加载器
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    # 获取一个样本来确定图像尺寸
    sample_data = next(iter(test_loader))
    moving_sample = sample_data[0]
    img_size = moving_sample.shape[2:]
    print(f"图像尺寸: {img_size}")
    
    # 重新创建数据加载器以确保从头开始
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    # 初始化模型
    model = VoxelMorph2D(img_size=img_size)
    
    # 加载最佳模型
    model_path = 'VoxelMorph/models/best_model.pth'
    print(f'加载最佳模型: {model_path}')
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # 获取数据
            moving = data[0].cuda()
            fixed = data[1].cuda()
            
            # 前向传播
            warped, flow = model(moving, fixed)
            
            # 保存结果图像
            # 转换为numpy数组并缩放到[0, 255]
            moving_np = moving.detach().cpu().numpy()[0, 0] * 255
            fixed_np = fixed.detach().cpu().numpy()[0, 0] * 255
            warped_np = warped.detach().cpu().numpy()[0, 0] * 255
            
            # 创建形变场的可视化
            flow_np = flow.detach().cpu().numpy()[0]
            flow_magnitude = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
            flow_magnitude = (flow_magnitude / flow_magnitude.max() * 255).astype(np.uint8)
            
             # 保存形变后的浮动图像到moved文件夹
            # 保存为灰度图像
            warped_img = Image.fromarray(warped_np.astype(np.uint8), mode='L')
            warped_img.save(os.path.join(moved_dir, f'moved_{idx}.png'))

            # 创建结果图像 - 简化为只显示三幅主要图像
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(fixed_np, cmap='gray')
            plt.title('固定图像')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(moving_np, cmap='gray')
            plt.title('浮动图像')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(warped_np, cmap='gray')
            plt.title('形变后的浮动图像')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f'result_{idx}.png'), bbox_inches='tight')
            plt.close()
            
            print(f'已处理并保存图像对 {idx+1}/{len(test_loader)}')


if __name__ == '__main__':
    # GPU配置
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('GPU数量: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('当前使用: ' + torch.cuda.get_device_name(GPU_iden))
    print('GPU是否可用? ' + str(GPU_avai))
    main()