import glob
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
import random
from PIL import Image
import warnings

# 导入自定义模块
from vxm2d import VoxelMorph2D, SpatialTransformer2D
import losses2d
import utils2d
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
    torch.backends.cudnn.benchmark = True

same_seeds(24)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def make_dirs(model_dir, log_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    """Adjust learning rate using polynomial decay"""
    lr = init_lr * (1 - epoch / max_epoch) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train():
    # 训练参数
    lr = 0.0001     # 学习率
    alpha = 1.0     # 梯度损失权重
    sim_loss = "mind" # 相似性损失函数
    gpu = 0         # GPU ID
    
    # 训练参数设置
    epoch_start = 0
    max_epoch = 300  # 训练轮数
    batch_size = 1  # 批次大小
    
    # 数据目录
    moving_dir = r'E:\RDP\data\train_1\moving'
    fixed_dir = r'E:\RDP\data\train_1\fixed'
    
    # 创建保存目录
    model_dir = 'VoxelMorph/models/'
    log_dir = 'VoxelMorph/logs/'
    make_dirs(model_dir, log_dir)
    
    # 指定GPU
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    # 日志文件
    log_name = f"{max_epoch}_{lr}_{alpha}_paired"
    print("log_name: ", log_name)
    f = open(os.path.join(log_dir, log_name + ".txt"), "w")
    
    # 创建数据集和数据加载器 - 使用ImageDataset2D类进行配对图像训练
    dataset = datasets2d.ImageDataset2D(moving_dir, fixed_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 获取一个样本来确定图像尺寸
    sample_moving, sample_fixed = next(iter(dataloader))
    img_size = sample_fixed.shape[2:]
    print(f"图像尺寸: {sample_fixed.shape}")
    
    # 重新创建数据加载器以确保从头开始
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"数据集大小: {len(dataset)}对图像")
    
    # 创建模型
    model = VoxelMorph2D(img_size=img_size).to(device)
    STN = SpatialTransformer2D(img_size).to(device)
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    
    # 选择相似性损失函数
    if sim_loss == "ncc":
        sim_loss_fn = losses2d.NCC_2d()
    else:  
        sim_loss_fn = losses2d.MIND_2D_loss()
        
    grad_loss_fn = losses2d.Grad2d(penalty='l2')
    
    # 最佳损失值（用于保存最佳模型）
    best_loss = float('inf')
    best_model_state = None  # 用于保存最佳模型的状态
    
     # 添加余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=lr*0.01)

    # 训练循环
    for epoch in range(epoch_start, max_epoch):
        print(f'\n开始第 {epoch+1}/{max_epoch} 轮训练')
        model.train()
        
        # 调整学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.8f}")

        # 用于记录平均损失
        loss_all = utils2d.AverageMeter()
        sim_loss_all = utils2d.AverageMeter()
        grad_loss_all = utils2d.AverageMeter()
        
        for i, (moving, fixed) in enumerate(dataloader):
                
            # 将数据移到设备上
            fixed = fixed.to(device).float()
            moving = moving.to(device).float()
            
            # 运行模型
            warped, flow = model(moving, fixed)
            
            # 计算损失
            sim_loss_val = sim_loss_fn(warped, fixed)
            grad_loss_val = grad_loss_fn(flow, None)
            loss = sim_loss_val + alpha * grad_loss_val
            
             # 检查损失是否异常
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告：检测到异常损失值 {loss.item()}，跳过此批次")
                continue

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            # 更新平均损失
            loss_all.update(loss.item())
            sim_loss_all.update(sim_loss_val.item())
            grad_loss_all.update(grad_loss_val.item())
            
       # 记录到日志文件
        print(f"{epoch+1}, {loss_all.avg:.6f}, {sim_loss_all.avg:.6f}, {grad_loss_all.avg:.6f}", file=f)
        
        # 每个epoch结束后打印平均损失
        print(f"Epoch {epoch+1} 平均损失: {loss_all.avg:.6f}, 相似性损失: {sim_loss_all.avg:.6f}, 梯度损失: {grad_loss_all.avg:.6f}")
        
        # 更新学习率调度器
        scheduler.step()

        # 检查是否是最佳模型
        if loss_all.avg < best_loss:
            best_loss = loss_all.avg
            best_model_state = model.state_dict().copy()  # 保存最佳模型状态
            print(f"发现新的最佳模型，损失: {best_loss:.6f}")
    
    # 训练结束后，只保存最佳模型
    if best_model_state is not None:
        print(f"保存最佳模型，最终损失: {best_loss:.6f}")
        torch.save({
            'epoch': max_epoch,
            'state_dict': best_model_state,
            'best_loss': best_loss,
        }, os.path.join(model_dir, f'best_model_{sim_loss}loss{best_loss:.4f}.pth'))
        
        # 同时保存一个固定名称的版本，方便后续使用
        torch.save(best_model_state, os.path.join(model_dir, 'best_model.pth'))
    
    f.close()
    print("训练完成！")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()