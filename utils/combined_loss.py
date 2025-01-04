''' This module defines a combination loss function . '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module): # 计算速度慢
    def __init__(self, threshold=1000,  mse_overweight=1.0,  mse_norm_coef = 1200000, class_norm_coef = 1.25):
        """
        初始化 CombinedLoss 类。

        Args:
            threshold (float): HIC 权重分配的阈值。
            mse_overweight (float): MSE 损失的过权重，超参数！
            mse_norm_coef (float): MSE 损失的归一化因子。
            class_norm_coef (float): 分类损失的归一化因子。
        """
        super(CombinedLoss, self).__init__()
        self.threshold = threshold
        self.ktanh = 0.01
        self.ksigmoid = 0.004
        self.mse_overweight = mse_overweight
        self.mse_norm_coef = mse_norm_coef
        self.class_norm_coef = class_norm_coef

    @staticmethod
    def AIS_prob_cal(HIC):
        """
        计算 HIC 到 AIS 分类概率的映射

        Args:
            HIC (torch.Tensor): 输入的 HIC 值张量，形状为 [batch_size]。

        Returns:
            torch.Tensor: 输出的 AIS 概率 logits, 形状为 [batch_size, 6]。
        """
        _HIC = torch.clamp(HIC, min=5.0, max=2000.0)

        hic = torch.zeros((_HIC.size(0), 5), device=HIC.device)
        hic[:, 0] = 1.0 / (1.0 + torch.exp(1.54 + 200.0 / _HIC - 0.00650 * _HIC))  # P(AIS≥1)
        hic[:, 1] = 1.0 / (1.0 + torch.exp(2.49 + 200.0 / _HIC - 0.00483 * _HIC))  # P(AIS≥2)
        hic[:, 2] = 1.0 / (1.0 + torch.exp(3.39 + 200.0 / _HIC - 0.00372 * _HIC))  # P(AIS≥3)
        hic[:, 3] = 1.0 / (1.0 + torch.exp(4.90 + 200.0 / _HIC - 0.00351 * _HIC))  # P(AIS≥4)
        hic[:, 4] = 1.0 / (1.0 + torch.exp(7.82 + 200.0 / _HIC - 0.00429 * _HIC))  # P(AIS≥5)

        ais_prob = torch.zeros((_HIC.size(0), 6), device=HIC.device)
        ais_prob[:, 0] = 1.0 - hic[:, 0]                   # P(AIS=0)
        ais_prob[:, 1] = hic[:, 0] - hic[:, 1]             # P(AIS=1)
        ais_prob[:, 2] = hic[:, 1] - hic[:, 2]             # P(AIS=2)
        ais_prob[:, 3] = hic[:, 2] - hic[:, 3]             # P(AIS=3)
        ais_prob[:, 4] = hic[:, 3] - hic[:, 4]             # P(AIS=4)
        ais_prob[:, 5] = hic[:, 4]                         # P(AIS=5)

        return ais_prob

    def weighted_function(self, x):
        """
        HIC 权重分配函数，分别生成分类和 MSE 损失的权重。

        Args:
            x (torch.Tensor): HIC 值张量，形状为 [batch_size]。

        Returns:
            torch.Tensor: 分类损失权重和 MSE 损失权重 [batch_size]。
        """
        tanh_part = 0.5 * (torch.tanh(self.ktanh * (x - self.threshold)) + 1)
        sigmoid_part = torch.sigmoid(self.ksigmoid * (x - self.threshold))
        y_class = torch.where(x <= self.threshold, tanh_part, sigmoid_part)
        y_mse = 1 - y_class
        return y_class, y_mse

    @staticmethod
    def cross_entropy_loss_from_prob(pred_ais_probs, true_ais):
        """
        计算基于经验概率的交叉熵损失。

        Args:
            pred_ais_probs (torch.Tensor): 经验概率，形状为 [batch_size, num_classes=6]。
            true_ais (torch.Tensor): 实际的类别标签，形状为 [batch_size]。

        Returns:
            batch_loss(torch.Tensor): 样本级的交叉熵损失，形状为 [batch_size]。
        """
        true_class_probs = pred_ais_probs[torch.arange(true_ais.size(0)), true_ais]
        # loss = -log(P(AIS_i=AIS_ture_i))
        batch_loss = -torch.log(true_class_probs + 1e-12)  # 防止 log(0) 数值溢出
        return batch_loss  # 返回每个样本的损失

    def forward(self, pred_hic, true_hic, true_ais):
        """
        计算综合损失。

        Args:
            pred_hic (torch.Tensor): 模型预测的 HIC 值，形状为 [batch_size]。
            true_hic (torch.Tensor): 实际 HIC 值，形状为 [batch_size]。
            true_ais (torch.Tensor): 实际 AIS 分类标签，形状为 [batch_size]。

        Returns:
            torch.Tensor: 综合损失值。
            torch.Tensor: 未样本加权的 MSE 损失（已归一化）。
            torch.Tensor: 未样本加权的分类损失（已归一化）。
        """
        # 调用权重函数计算权重
        y_class, y_mse = self.weighted_function(true_hic) # 形状均为 [batch_size]

        # 计算样本加权后的分类损失
        pred_ais_probs = self.AIS_prob_cal(pred_hic) # 形状为 [batch_size, 6]
        class_losses = self.cross_entropy_loss_from_prob(pred_ais_probs, true_ais) # 对样本加权后取均值
        weighted_class_loss  = (class_losses * y_class).mean()  # 对样本加权后取均值

        # 计算样本加权后 MSE 损失
        mse_losses = F.mse_loss(pred_hic, true_hic, reduction='none')  # 样本级 MSE 损失
        weighted_mse_loss  = (mse_losses * y_mse).mean()        # 对样本加权后取均值
        
        # 综合损失 mse_overweight 控制 MSE 损失的相对重要性
        combined_loss = self.mse_overweight * weighted_mse_loss  / self.mse_norm_coef + weighted_class_loss  / self.class_norm_coef

        return combined_loss, mse_losses.mean() / self.mse_norm_coef, class_losses.mean() / self.class_norm_coef

if __name__ == '__main__':
    import numpy as np
    # Test the CombinedLoss class
    AIS_prob = np.array([0.477, 0.124, 0.122, 0.092, 0.051, 0.134], dtype=np.float32) 
    print(-np.log(np.sum(AIS_prob ** 2))) # 1.249
    criterion = CombinedLoss(class_norm_coef=-np.log(np.sum(AIS_prob ** 2)))   
    pred_hic = torch.tensor([0, 100, 1000.0, 2600.0], dtype=torch.float32)
    true_hic = torch.tensor([0, 300, 900.0, 1500.0], dtype=torch.float32)
    true_ais = torch.tensor([0, 1, 2, 5], dtype=torch.long)
    loss = criterion(pred_hic, true_hic, true_ais)
    print(loss)
    #loss.backward()
