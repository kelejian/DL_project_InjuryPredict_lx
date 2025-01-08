''' This module defines a weighted loss function . '''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch

def Piecewise_linear(y_true, y_pred, weight_add_mid=1.0):
    """
    计算分段线性权重增加量。

    参数:
        y_true (torch.Tensor): 真实标签，形状为 (B,)。
        weight_add_mid (float): 中间区间 [150, 1600] 的权重增加量

    返回:
        weight_adds (torch.Tensor): 权重增加量，形状与 y_true 相同。
    """
    # 初始化权重增加量为 0
    weight_adds = torch.zeros_like(y_true)
    # 区间 1: 0 <= y <= 100，权重增加量为 0
    mask_0_100 = (y_true >= 0) & (y_true <= 100)
    weight_adds[mask_0_100] = 0
    # 区间 2: 100 < y < 150，线性递增，斜率为 weight_add_mid / 50
    mask_100_150 = (y_true > 100) & (y_true < 150)
    weight_adds[mask_100_150] = (weight_add_mid / 50) * (y_true[mask_100_150] - 100)
    # 区间 3: 150 <= y <= 1600，权重增加量为 weight_add_mid
    mask_150_1600 = (y_true >= 150) & (y_true <= 1600)
    weight_adds[mask_150_1600] = weight_add_mid
    # 区间 4: 1600 < y < 1800，线性递减，斜率为 -weight_add_mid / 200
    mask_1600_1800 = (y_true > 1600) & (y_true < 1800)
    weight_adds[mask_1600_1800] = weight_add_mid + (-weight_add_mid / 200) * (y_true[mask_1600_1800] - 1600)
    # 区间 5: 1800 <= y <= 2500，权重增加量为 0
    mask_1800_2500 = (y_true >= 1800) & (y_true <= 2500)
    weight_adds[mask_1800_2500] = 0
    # 区间 6: y > 2500，权重减少
    mask_2500_plus = y_true > 2500
    weight_adds[mask_2500_plus] = -1 + torch.exp(-5e-4 * (y_true[mask_2500_plus] - 2500))

    # 最后增加y_pred<0的权重惩罚, 线性递增，斜率绝对值为 weight_add_mid / 50
    mask_pred_neg = y_pred < 0
    weight_adds[mask_pred_neg] = (weight_add_mid / 50) * (-y_pred[mask_pred_neg])

    return weight_adds

class weighted_loss(nn.Module): 
    """
    加权损失函数, 对分类错误的样本和中间HIC值的样本赋予更大的权重。
    """
    def __init__(self, base_loss="mse", weight_factor_classify=1.1, weight_factor_sample=1.0, y_transform=None):
        super(weighted_loss, self).__init__()
        '''
        base_loss: str, 基础损失函数，可以是 'mse' 或 'mae'
        weight_factor_classify: float, 基于AIS-6C分类的样本权重系数, 分类错误的样本对应的loss会被放大
        weight_factor_sample: float, 中间HIC值的样本权重系数, 中间HIC值的样本对应的loss会被放大
        '''

        # base loss 要么是 mse 要么是mae
        if base_loss == "mse":
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss == "mae":
            self.base_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError("base_loss should be 'mse' or 'mae'.")
        
        self.weight_factor_classify = weight_factor_classify
        self.weight_factor_sample = weight_factor_sample
        self.y_transform = y_transform

    def AIS_cal(self, HIC):
        """
        计算 AIS 等级(Torch 版本)
        
        参数:
            HIC (torch.Tensor): 输入的 HIC 值，(B,)
        返回:
            AIS (torch.Tensor): 计算得到的 AIS 等级，(B,)
        """
        # 限制 HIC 范围，防止数值不稳定
        HIC = torch.clamp(HIC, 1, 2000)
        
        # 经验概率阈值
        threshold = 0.2

        # 定义常量和系数
        coefficients = torch.tensor([
            [1.54, 0.00650],  # P(AIS≥1)
            [2.49, 0.00483],  # P(AIS≥2)
            [3.39, 0.00372],  # P(AIS≥3)
            [4.90, 0.00351],  # P(AIS≥4)
            [7.82, 0.00429]   # P(AIS≥5)
        ], device=HIC.device)

        # 计算 P(AIS≥n) 的概率（向量化计算）
        c1 = coefficients[:, 0].unsqueeze(1)  # 系数1，形状为 (5, 1)
        c2 = coefficients[:, 1].unsqueeze(1)  # 系数2，形状为 (5, 1)
        HIC_inv = 200 / HIC.unsqueeze(0)  # HIC 的倒数部分，形状为 (1, B)

        # 计算所有 P(AIS≥n)
        hic_prob = 1.0 / (1.0 + torch.exp(c1 + HIC_inv - c2 * HIC.unsqueeze(0)))  # 形状为 (5, B)

        # 确定 AIS 等级
        AIS = torch.sum(hic_prob >= threshold, dim=0)  # 形状为 (B,)

        return AIS
    
    def weighted_function(self, pred_hic, true_hic):
        """
        样本权重函数。根据AIC-6C分类准确率和HIC值区间范围计算不同样本的权重
        """
        if self.y_transform is not None:
            pred_hic = self.y_transform.inverse(pred_hic)
            true_hic = self.y_transform.inverse(true_hic)

            pred_ais = self.AIS_cal(pred_hic)
            true_ais = self.AIS_cal(true_hic)
            weights_classify = self.weight_factor_classify ** torch.abs(pred_ais - true_ais) # 根据AIS-6C分类准确率计算样本权重, 分类错误的样本权重会被放大
            # 预测为负值的样本权重也会被放大
            weight_adds = torch.zeros_like(true_hic)
            weight_adds[pred_hic < 0] = (self.weight_factor_sample / 25) * (-pred_hic[pred_hic < 0])
            weights_mid = 1.0 + weight_adds
        
        elif self.y_transform is None:
            pred_ais = self.AIS_cal(pred_hic)
            true_ais = self.AIS_cal(true_hic)
            weights_classify = self.weight_factor_classify ** torch.abs(pred_ais - true_ais)# 根据AIS-6C分类准确率计算样本权重, 分类错误的样本权重会被放大
            weights_mid = 1.0 + Piecewise_linear(true_hic, pred_hic, self.weight_factor_sample) # 根据HIC值区间范围计算样本权重, 中间HIC值的样本权重会被放大, 预测为负值的样本权重也会被放大

        return weights_classify * weights_mid

    def forward(self, pred_hic, true_hic):
        """
        计算加权损失。
        """
        weights = self.weighted_function(pred_hic, true_hic)
        weighted_losses = self.base_loss(pred_hic, true_hic) * weights
        
        return weighted_losses.mean()


if __name__ == '__main__':
    import numpy as np
    # Test the CombinedLoss class
    pred_hic = torch.tensor([-50, 100, 1000.0, 5000.0], dtype=torch.float32)
    true_hic = torch.tensor([0, 50, 900.0, 10000.0], dtype=torch.float32)
    from dataset_prepare import SigmoidTransform
    transform = SigmoidTransform(0, 2500)
    transform = None
    # pred_hic = transform.forward(pred_hic)
    # true_hic = transform.forward(true_hic)
    criterion = weighted_loss(weight_factor_classify=1.50, weight_factor_sample=2.0, y_transform=transform)  
    criterion_base = nn.MSELoss()
    loss = criterion(pred_hic, true_hic)
    loss_base = criterion_base(pred_hic, true_hic)
    print(loss, loss_base)
    # 示例输入

    #loss.backward()
