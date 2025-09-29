''' This module defines a weighted loss function for multiple injury criteria. '''
import torch
import torch.nn as nn
import numpy as np
try:
    from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck  # 作为包导入时使用
except ImportError:
    from AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck   # 直接运行时使用

def Piecewise_linear(y_true, y_pred, params, weight_add_mid=1.0):
    """
    通用分段线性权重增加量计算函数。

    参数:
        y_true (torch.Tensor): 真实标签, (B,)
        y_pred (torch.Tensor): 预测值, (B,)
        params (dict): 包含 'a', 'b', 'c', 'd', 't' 的超参数字典
        weight_add_mid (float): 中间区间的权重增加量
    返回:
        weight_adds (torch.Tensor): 权重增加量, (B,)
    """
    a, b, c, d, t = params['a'], params['b'], params['c'], params['d'], params['t']
    
    weight_adds = torch.zeros_like(y_true)
    
    # 区间 1: 0 <= y < a，线性递增至 weight_add_mid
    mask = (y_true >= 0) & (y_true < a)
    if a > 0:
        weight_adds[mask] = (weight_add_mid / a) * y_true[mask]

    # 区间 2: a <= y <= b，权重增加量为 weight_add_mid
    mask = (y_true >= a) & (y_true <= b)
    weight_adds[mask] = weight_add_mid

    # 区间 3: b < y < c，线性递减至 0
    mask = (y_true > b) & (y_true < c)
    if c > b:
        weight_adds[mask] = weight_add_mid - (weight_add_mid / (c - b)) * (y_true[mask] - b)

    # 区间 4: c <= y <= d，权重增加量为 0
    mask = (y_true >= c) & (y_true <= d)
    weight_adds[mask] = 0
    
    # 区间 5: y > d，指数递减至 -1
    mask = y_true > d
    if d > 0:
        # 根据 y_true=2d 时 weight_adds=t 反解出衰减系数 k
        # t = -1 + exp(-k * (2d - d)) => t + 1 = exp(-k*d) => k = -ln(t+1)/d
        if t > -1: # 确保 log 的输入为正
             k = -np.log(t + 1) / d
             weight_adds[mask] = -1 + torch.exp(-k * (y_true[mask] - d))
        else: # 如果 t <= -1, 使用一个默认的衰减
             weight_adds[mask] = -1 
             print("Warning: t should be > -1 for exponential decay. Using default weight -1 for y > d.")
   
    # 最后增加 y_pred < 0 的权重惩罚
    mask_pred_neg = y_pred < 0
    weight_adds[mask_pred_neg] += weight_add_mid

    return weight_adds

class weighted_loss(nn.Module): 
    def __init__(self, base_loss="mae", weight_factor_classify=1.1, weight_factor_sample=1.0, loss_weights=(1.0, 1.0, 1.0)):
        super(weighted_loss, self).__init__()
        '''
        base_loss: str, 基础损失函数 'mse' 或 'mae'
        weight_factor_classify: float, 基于AIS分类的样本权重系数
        weight_factor_sample: float, 中间值样本的权重系数, 用于分段线性函数
        loss_weights: tuple, (w_hic, w_dmax, w_nij) 三个任务损失的权重
        '''

        if base_loss == "mse":
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss == "mae":
            self.base_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError("base_loss should be 'mse' or 'mae'.")
        
        self.weight_factor_classify = weight_factor_classify
        self.weight_factor_sample = weight_factor_sample
        self.loss_weights = loss_weights
        
        # --- 为每个损伤部位定义分段加权超参数 ---
        self.params_head = {'a': 60, 'b': 1600, 'c': 1800, 'd': 2000, 't': -0.5}
        self.params_chest = {'a': 5.0, 'b': 175, 'c': 195, 'd': 220, 't': -0.5}
        self.params_neck = {'a': 0.3, 'b': 2.0, 'c': 2.3, 'd': 2.75, 't': -0.5}
 
    def weighted_function(self, pred, true, injury_type):
        """
        样本权重函数。根据AIS分类准确率和损伤值区间范围计算不同样本的权重。
        """
        # --- 将PyTorch张量转换为Numpy数组以调用外部函数 ---
        # .detach()确保此操作脱离计算图, .cpu()移至CPU, .numpy()转换为Numpy数组
        pred_np = pred.detach().cpu().numpy()
        true_np = true.detach().cpu().numpy()

        if injury_type == 'head':
            # 使用Numpy函数计算AIS
            pred_ais_np = AIS_cal_head(pred_np)
            true_ais_np = AIS_cal_head(true_np)
            # 使用通用分段函数计算样本区间权重
            weights_mid = 1.0 + Piecewise_linear(true, pred, self.params_head, self.weight_factor_sample)
        elif injury_type == 'chest':
            pred_ais_np = AIS_cal_chest(pred_np)
            true_ais_np = AIS_cal_chest(true_np)
            weights_mid = 1.0 + Piecewise_linear(true, pred, self.params_chest, self.weight_factor_sample)
        elif injury_type == 'neck':
            pred_ais_np = AIS_cal_neck(pred_np)
            true_ais_np = AIS_cal_neck(true_np)
            weights_mid = 1.0 + Piecewise_linear(true, pred, self.params_neck, self.weight_factor_sample)
        else:
            raise ValueError(f"Unknown injury_type: {injury_type}")

        # --- 将Numpy结果转换回PyTorch张量 ---
        # 确保张量设备与输入张量一致，以便进行后续计算
        pred_ais = torch.tensor(pred_ais_np, device=pred.device)
        true_ais = torch.tensor(true_ais_np, device=true.device)

        # 根据AIS分类准确率计算样本权重, 分类错误的样本权重会被放大
        weights_classify = self.weight_factor_classify ** torch.abs(pred_ais.float() - true_ais.float())
        
        # 最终权重是分类权重和样本区间权重的乘积
        return weights_classify * weights_mid

    def forward(self, pred, true):
        """
        计算加权多任务损失。
        Args:
            pred (torch.Tensor): 模型的预测输出, 形状为 (B, 3)。列顺序: HIC, Dmax, Nij。
            true (torch.Tensor): 真实标签, 形状为 (B, 3)。列顺序: HIC, Dmax, Nij。
        Returns:
            total_loss (torch.Tensor): 加权总损失。为标量张量。
        """
        # --- 分离不同损伤的预测和标签 ---
        pred_hic, pred_dmax, pred_nij = pred[:, 0], pred[:, 1], pred[:, 2]
        true_hic, true_dmax, true_nij = true[:, 0], true[:, 1], true[:, 2]

        # --- 分别计算每个任务的加权损失 ---
        # HIC Loss
        weights_hic = self.weighted_function(pred_hic, true_hic, 'head')
        loss_hic = (self.base_loss(pred_hic, true_hic) * weights_hic).mean()

        # Dmax Loss
        weights_dmax = self.weighted_function(pred_dmax, true_dmax, 'chest')
        loss_dmax = (self.base_loss(pred_dmax, true_dmax) * weights_dmax).mean()

        # Nij Loss
        weights_nij = self.weighted_function(pred_nij, true_nij, 'neck')
        loss_nij = (self.base_loss(pred_nij, true_nij) * weights_nij).mean()
        
        # --- 使用预设权重合并总损失 ---
        w_hic, w_dmax, w_nij = self.loss_weights
        total_loss = w_hic * loss_hic + w_dmax * loss_dmax + w_nij * loss_nij
        
        return total_loss


if __name__ == '__main__':
    # Test the weighted_loss class
    pred = torch.tensor([
        [100.0, 10.0, 0.8],     # 预测1
        [1800.0, 220.0, 3.0],   # 预测2
        [-50.0, -5.0, -0.2]     # 预测3 (含负值)
    ], dtype=torch.float32)
    true = torch.tensor([
        [50.0, 5.0, 0.5],       # 真实1
        [1700.0, 210.0, 2.5],   # 真实2
        [10.0, 2.0, 0.1]        # 真实3
    ], dtype=torch.float32)

    # 初始化损失函数，可以指定各部分损失的权重
    criterion = weighted_loss(
        base_loss='mae', 
        weight_factor_classify=1.5, 
        weight_factor_sample=2.0,
        loss_weights=(1.0, 0.8, 1.2) # 示例权重：HIC权重1.0, Dmax权重0.8, Nij权重1.2
    )
    
    loss = criterion(pred, true)
    print("Total Weighted MAE Loss:", loss.item())

    # 对比基础的MAE Loss
    criterion_base = nn.L1Loss()
    loss_base = criterion_base(pred, true)
    print("Base MAE Loss:", loss_base.item())