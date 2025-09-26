# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os, json
import time
from datetime import datetime
import torch
import random
import warnings
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from utils import models
from utils.weighted_loss import weighted_loss
from utils.dataset_prepare import SigmoidTransform, CrashDataset, AIS_cal

from utils.set_random_seed import set_random_seed
set_random_seed()

def train_student(model, loader, optimizer, criterion, device):
    """
    参数:
        model: 学生模型实例。
        loader: 数据加载器。
        optimizer: 优化器。
        criterion: 损失函数。
        device: GPU 或 CPU。
    """
    model.train()
    loss_batch = []
    for _, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, _ in loader:
        # 将数据移动到设备
        batch_x_att_continuous = batch_x_att_continuous.to(device)
        batch_x_att_discrete = batch_x_att_discrete.to(device)
        batch_y_HIC = batch_y_HIC.to(device)

        # 前向传播
        batch_pred_HIC, encoder_output, decoder_output = model(batch_x_att_continuous, batch_x_att_discrete)

        # 计算损失
        loss = criterion(batch_pred_HIC, batch_y_HIC)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        loss_batch.append(loss.item())

    return np.mean(loss_batch)

def valid_student(model, loader, criterion, device):
    """
    参数:
        model: 学生模型实例。
        loader: 数据加载器。
        criterion: 损失函数。
        device: GPU 或 CPU。
    返回:
        avg_loss: 学生模型的回归损失（基于 criterion 计算）。
        accuracy: 验证集的分类准确率。
        mae: 验证集的平均绝对误差。
        rmse: 验证集的均方根误差。
    """
    model.eval()
    loss_batch = []
    all_HIC_preds = []
    all_HIC_trues = []
    all_AIS_trues = []

    with torch.no_grad():
        for _, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, batch_y_AIS in loader:
            # 将数据移动到设备
            batch_x_att_continuous = batch_x_att_continuous.to(device)
            batch_x_att_discrete = batch_x_att_discrete.to(device)
            batch_y_HIC = batch_y_HIC.to(device)
            batch_y_AIS = batch_y_AIS.to(device)

            # 前向传播
            batch_pred_HIC, _, _ = model(batch_x_att_continuous, batch_x_att_discrete)

            # 计算损失
            loss = criterion(batch_pred_HIC, batch_y_HIC)
            loss_batch.append(loss.item())

            # 记录预测值和真实值
            all_HIC_preds.append(batch_pred_HIC.cpu().numpy())
            all_HIC_trues.append(batch_y_HIC.cpu().numpy())
            all_AIS_trues.append(batch_y_AIS.cpu().numpy())

    # 计算平均损失
    avg_loss = np.mean(loss_batch)

    # 拼接所有 batch 的结果
    HIC_preds = np.concatenate(all_HIC_preds)
    HIC_trues = np.concatenate(all_HIC_trues)
    AIS_trues = np.concatenate(all_AIS_trues)

    # 检查 HIC_preds 和 HIC_trues 是否包含 inf 或异常值
    if np.isinf(HIC_preds).any() or np.isinf(HIC_trues).any():
        print("**Warning: HIC_preds or HIC_trues contains inf values. Replacing inf with large finite values.**")
        HIC_preds = np.nan_to_num(HIC_preds, nan=0.0, posinf=1e4, neginf=-1e4)
        HIC_trues = np.nan_to_num(HIC_trues, nan=0.0, posinf=1e4, neginf=-1e4)

    # 计算准确率
    AIS_preds = AIS_cal(HIC_preds)
    accuracy = 100. * (1 - np.count_nonzero(AIS_preds - AIS_trues) / len(AIS_trues))

    # 计算MAE, RMSE
    HIC_preds[HIC_preds > 2500] = 2500 # 规定上界，过高的HIC值(>2000基本就危重伤)不具有实际意义
    HIC_trues[HIC_trues > 2500] = 2500  
    mae = mean_absolute_error(HIC_trues, HIC_preds)
    rmse = root_mean_squared_error(HIC_trues, HIC_preds)

    return avg_loss, accuracy, mae, rmse


if __name__ == "__main__":
    ''' Train the student model without knowledge distillation. '''
    from torch.utils.tensorboard import SummaryWriter

    # 创建独立文件夹保存本次运行结果
    current_time = datetime.now().strftime("%m%d%H%M")  # 月日时分
    run_dir = os.path.join("./runs", f"StudentModel_Baseline_{current_time}")
    os.makedirs(run_dir, exist_ok=True)

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=run_dir)
    
    ############################################################################################
    ############################################################################################
    # 定义优化相关的超参数
    Epochs = 500
    Batch_size = 512
    Learning_rate = 0.024  # 初始学习率
    Learning_rate_min = 2e-7  # 余弦退火最小学习率
    weight_decay = 6e-4  # L2 正则化系数
    Patience = 888  # 早停等待轮数
    base_loss = "mae"  # 或 "mae"
    weight_factor_classify = 3.0  # 加权损失函数的系数1
    weight_factor_sample = 0.6  # 加权损失函数的系数2
    # 定义模型相关的超参数
    num_layers_of_mlpE = 4  # MLP 编码器的层数
    num_layers_of_mlpD = 4  # MLP 解码器的层数
    mlpE_hidden = 160  # MLP 编码器的隐藏层维度
    mlpD_hidden = 128  # MLP 解码器的隐藏层维度
    encoder_output_dim = 96  # 编码器输出特征维度
    decoder_output_dim = 32  # 解码器输出特征维度
    dropout = 0.15  # Dropout 概率
    ############################################################################################
    ############################################################################################

    # 加载数据集对象
    dataset = CrashDataset()
    # 从data文件夹直接加载数据集
    train_dataset = torch.load("./data/train_dataset.pt")
    val_dataset = torch.load("./data/val_dataset.pt")

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化学生模型
    model = models.StudentModel(
        num_classes_of_discrete=dataset.num_classes_of_discrete,
        num_layers_of_mlpE=num_layers_of_mlpE,
        num_layers_of_mlpD=num_layers_of_mlpD,
        mlpE_hidden=mlpE_hidden,
        mlpD_hidden=mlpD_hidden,
        encoder_output_dim=encoder_output_dim,
        decoder_output_dim=decoder_output_dim,
        dropout=dropout
    ).to(device)

    # 定义损失函数、优化器和学习率调度器
    criterion = weighted_loss(base_loss, weight_factor_classify, weight_factor_sample)
    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)

    # 初始化变量
    LossCurve_val, LossCurve_train = [], []
    Best_accu = 0
    Best_mae = 1e5
    Best_rmse = 1e5

    # 主训练循环
    for epoch in range(Epochs):
        epoch_start_time = time.time()

        # 训练模型
        train_loss = train_student(model, train_loader, optimizer, criterion, device)
        LossCurve_train.append(train_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Train Loss: {train_loss:.3f}")

        # 验证模型
        val_loss, val_accuracy, val_mae, val_rmse = valid_student(model, val_loader, criterion, device)
        LossCurve_val.append(val_loss)
        print(f"            | Val Loss: {val_loss:.3f} | Val Accuracy: {val_accuracy:.1f}% | MAE: {val_mae:.1f} | RMSE: {val_rmse:.1f}")

        # 学习率调整
        scheduler.step()

        # TensorBoard 记录
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_accuracy, epoch)
        writer.add_scalar("Mae/Val", val_mae, epoch)
        writer.add_scalar("Rmse/Val", val_rmse, epoch)

        # 保存分类准确率最佳的模型
        if val_accuracy > Best_accu:
            Best_accu = val_accuracy
            if Best_accu > 50:
                torch.save(model.state_dict(), os.path.join(run_dir, "student_best_accu.pth"))
                print(f"Best model saved with val accuracy: {Best_accu:.2f}%")

        # 保存 MAE 最佳的模型
        if val_mae < Best_mae:
            Best_mae = val_mae
            if Best_mae < 200:
                torch.save(model.state_dict(), os.path.join(run_dir, "student_best_mae.pth"))
                print(f"Best model saved with val MAE: {Best_mae:.1f}")

        # 保存 RMSE 最佳的模型
        if val_rmse < Best_rmse:
            Best_rmse = val_rmse
            if Best_rmse < 300:
                torch.save(model.state_dict(), os.path.join(run_dir, "student_best_rmse.pth"))
                print(f"Best model saved with val RMSE: {Best_rmse:.1f}")

        # 早停逻辑
        if len(LossCurve_val) > Patience:
            recent_losses = LossCurve_val[-Patience:]
            if all(recent_losses[i] < recent_losses[i + 1] for i in range(len(recent_losses) - 1)): # 如果最近连续 Patience 轮 val loss 都上升
                print(f"Early Stop at epoch: {epoch+1}! Last val accuracy: {val_accuracy:.1f}%! Best val accuracy: {Best_accu:.1f}%")
                print(f"Last val MAE: {val_mae:.1f}! Best val MAE: {Best_mae:.1f}")
                print(f"Last val RMSE: {val_rmse:.1f}! Best val RMSE: {Best_rmse:.1f}")
                break

        print(f"            | Time: {time.time()-epoch_start_time:.2f}s")

    # 关闭 TensorBoard
    writer.close()

    # 保存超参数和关键结果到 JSON 文件
    results = {
        "hyperparameters related to training": {
            "Epochs": Epochs,
            "Batch_size": Batch_size,
            "Learning_rate": float(Learning_rate),
            "Learning_rate_min": float(Learning_rate_min),
            "weight_decay": float(weight_decay),
            "Patience": Patience,
        },
        "hyperparameters related to model": {
            "num_layers_of_mlpE": num_layers_of_mlpE,
            "num_layers_of_mlpD": num_layers_of_mlpD,
            "mlpE_hidden": mlpE_hidden,
            "mlpD_hidden": mlpD_hidden,
            "encoder_output_dim": encoder_output_dim,
            "decoder_output_dim": decoder_output_dim,
            "dropout": float(dropout),
        },
        "results": {
            "final_epoch": epoch + 1,
            "Best_accu": float(Best_accu), 
            "Best_mae": float(Best_mae),  
            "Best_rmse": float(Best_rmse),  
            "last_val_accuracy": float(val_accuracy), 
            "last_val_mae": float(val_mae), 
            "last_val_rmse": float(val_rmse), 
        }
    }

    if isinstance(criterion, weighted_loss):
        results["hyperparameters related to training"]["base_loss"] = base_loss
        results["hyperparameters related to training"]["weight_factor_classify"] = weight_factor_classify
        results["hyperparameters related to training"]["weight_factor_sample"] = weight_factor_sample

    with open(os.path.join(run_dir, "TrainingRecord.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Training Finished!")