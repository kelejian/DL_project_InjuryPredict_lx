''' This module defines a weighted loss function . '''
import torch
import torch.nn as nn
import torch.nn.functional as F


class weighted_loss(nn.Module): 
    """
    基于AIS-6C分类标签和基础回归loss的加权损失函数, 对分类错误的样本赋予更大的权重。
    """
    def __init__(self, base_loss="mse", weighted_factor=1.1):
        super(weighted_loss, self).__init__()

        # base loss 要么是 mse 要么是mae
        if base_loss == "mse":
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss == "mae":
            self.base_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError("base_loss should be 'mse' or 'mae'.")
        
        self.weighted_factor = weighted_factor

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
        样本权重函数。计算不同样本的权重: weight = weighted_factor^abs(AIS_pred-AIS_true)

        """
        pred_ais = self.AIS_cal(pred_hic)
        true_ais = self.AIS_cal(true_hic)
        # 样本权重=weighted_factor^abs(AIS_pred-AIS_true)
        weights = self.weighted_factor ** torch.abs(pred_ais - true_ais)

        return weights

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
    pred_hic = torch.tensor([0, 100, 1000.0, 2600.0], dtype=torch.float32)
    true_hic = torch.tensor([0, 120, 900.0, 2200.0], dtype=torch.float32)

    criterion = weighted_loss(weighted_factor=1.10)  
    criterion_base = nn.MSELoss()
    loss = criterion(pred_hic, true_hic)
    loss_base = criterion_base(pred_hic, true_hic)
    print(loss, loss_base)
    #loss.backward()
