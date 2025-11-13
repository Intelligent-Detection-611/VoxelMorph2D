"""2D Losses for VoxelMorph"""

import math
import torch
import numpy as np
import torch.nn.functional as F


class Grad2d(torch.nn.Module):
    """
    2D gradient loss.
    """

    def __init__(self, penalty='l2', loss_mult=None):
        super(Grad2d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult  # 可选的损失函数系数，用于放大或者缩小损失权重

    def forward(self, y_pred, y_true):
        # 相邻两行的像素值差的绝对值
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        # 相邻两列的像素值差的绝对值
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        # 梯度惩罚类型，L1范数惩罚更鲁棒，对少数噪声点不敏感；L2范数更敏感，抑制剧烈变化
        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class NCC_2d(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss for 2D images.
    """

    def __init__(self, win=None):
        super(NCC_2d, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        ndims = 2  # 2D images

        # set window size
        win = [9, 9] if self.win is None else self.win

        # compute filters
        # 定义一个卷积核，大小为窗口大小即9×9
        sum_filt = torch.ones([1, 1, *win]).to(Ii.device)

        # 填充输入图像边缘，保持输出大小与输入相同
        # 如窗口大小为9时，pad_no为4，上下左右填充4像素，卷积核能完全覆盖输入图像边缘部分
        pad_no = math.floor(win[0] / 2)

        stride = (1, 1)
        padding = (pad_no, pad_no)

        # get convolution function
        conv_fn = F.conv2d

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # 计算局部窗口内各项和
        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)  # 窗口元素总数
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class MIND_2D_loss(torch.nn.Module):
    """
    MIND (Modality Independent Neighbourhood Descriptor) loss for 2D image registration.
    Adapted from 3D version for 2D images.
    
    Reference: Heinrich et al. "MIND: Modality independent neighbourhood descriptor 
    for multi-modal deformable registration." Medical image analysis 16.7 (2012): 1423-1435.
    """
    
    def __init__(self, win=None):
        super(MIND_2D_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        """计算成对距离的平方"""
        xx = (x ** 2).sum(dim=1).unsqueeze(2) #计算每个点的平方和，并增加维度用于传播
        yy = xx.permute(0, 2, 1) #转置xx用于成对距离计算
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x) #（x-y）^2 = x^2 + y^2 - 2xy
        dist[dist != dist] = 0  # 处理NaN值
        dist = torch.clamp(dist, 0.0, np.inf) #将负值截断为0，确保距离非负
        return dist

    def MINDSSC_2D(self, img, radius=2, dilation=2):
        """
        计算2D图像的MIND-SSC描述符
        
        Args:
            img: 输入2D图像 [batch, channel, height, width]
            radius: 邻域半径
            dilation: 膨胀系数
        """
        # 核大小
        kernel_size = radius * 2 + 1
        
        # 定义2D的4邻域模式 (上下左右)相对于3×3网格的中心点（1，1）
        four_neighbourhood = torch.Tensor([[0, 1],  # 上
                                          [1, 0],   # 左  
                                          [1, 2],   # 右
                                          [2, 1]]).long()  # 下
        
        # 计算成对距离的平方
        dist = self.pdist_squared(four_neighbourhood.t().unsqueeze(0)).squeeze(0)
        
        # 定义比较掩码
        x, y = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
        mask = ((x > y).view(-1) & (dist == 2).view(-1))
        
        # 构建卷积核
        idx_shift1 = four_neighbourhood.unsqueeze(1).repeat(1, 4, 1).view(-1, 2)[mask, :]
        idx_shift2 = four_neighbourhood.unsqueeze(0).repeat(4, 1, 1).view(-1, 2)[mask, :]
        
        # 创建2D卷积核 (对于2D情况，我们有6对比较)
        num_pairs = idx_shift1.shape[0]
        mshift1 = torch.zeros(num_pairs, 1, 3, 3).to(img.device)
        mshift2 = torch.zeros(num_pairs, 1, 3, 3).to(img.device)
        
        # 填充卷积核
        for i in range(num_pairs):
            mshift1[i, 0, idx_shift1[i, 0], idx_shift1[i, 1]] = 1
            mshift2[i, 0, idx_shift2[i, 0], idx_shift2[i, 1]] = 1
        
        # 填充层
        rpad1 = torch.nn.ReplicationPad2d(dilation)
        rpad2 = torch.nn.ReplicationPad2d(radius)
        
        # 计算patch-ssd
        conv1 = F.conv2d(rpad1(img), mshift1, dilation=dilation)
        conv2 = F.conv2d(rpad1(img), mshift2, dilation=dilation)
        ssd = F.avg_pool2d(rpad2((conv1 - conv2) ** 2), kernel_size, stride=1)
        
        # MIND方程
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, 
                              (mind_var.mean() * 0.001).item(), 
                              (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)
        
        return mind

    def forward(self, y_pred, y_true):
        """
        计算MIND损失
        
        Args:
            y_pred: 预测图像 [batch, channel, height, width]
            y_true: 真实图像 [batch, channel, height, width]
        
        Returns:
            MIND损失值
        """
        mind_pred = self.MINDSSC_2D(y_pred)
        mind_true = self.MINDSSC_2D(y_true)
        return torch.mean((mind_pred - mind_true) ** 2)