# -*- coding: utf-8 -*-
# 2024.12.19
import os, json
import time
from datetime import datetime
import torch
import random
import warnings
import numpy as np
from torch import nn
from torch.utils.data import  DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, root_mean_squared_error

from utils import models
from utils.dataset_prepare import SigmoidTransform, CrashDataset, AIS_cal, AIS_3_cal

#import wandb

warnings.filterwarnings('ignore')

# Define the random seed.
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train(model, loader, optimizer, criterion, device):
    """
    参数:
        model: 模型实例。
        loader: 数据加载器。
        optimizer: 优化器。
        criterion: 损失函数。
        device: GPU 或 CPU。
    """
    model.train()
    loss_batch = []
    for batch_x_acc, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, batch_y_AIS in loader:
        # 将数据移动到设备
        batch_x_acc = batch_x_acc.to(device)
        batch_x_att_continuous = batch_x_att_continuous.to(device)
        batch_x_att_discrete = batch_x_att_discrete.to(device)
        batch_y_HIC = batch_y_HIC.to(device)
        batch_y_AIS = batch_y_AIS.to(device)

        # 前向传播
        batch_pred_HIC, encoder_output, decoder_output = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)

        # 计算损失
        loss = criterion(batch_pred_HIC, batch_y_HIC)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        loss_batch.append(loss.item())

    return np.mean(loss_batch)

def valid(model, loader, criterion, device, y_transform=None):
    """
    验证函数。

    参数:
        model: 教师模型实例。
        loader: 数据加载器。
        criterion: 损失函数。
        device: GPU 或 CPU。
        y_transform: 数据集中的HIC标签变换对象。若数据集中的HIC标签没有进行变换则为None。
    """
    model.eval()
    loss_batch = []
    all_HIC_preds = []
    all_HIC_trues = []
    all_AIS_trues = []

    with torch.no_grad():
        for batch_x_acc, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, batch_y_AIS in loader:
            # 将数据移动到设备
            batch_x_acc = batch_x_acc.to(device)
            batch_x_att_continuous = batch_x_att_continuous.to(device)
            batch_x_att_discrete = batch_x_att_discrete.to(device)
            batch_y_HIC = batch_y_HIC.to(device)
            batch_y_AIS = batch_y_AIS.to(device)

            # 前向传播
            batch_pred_HIC, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)

            # 计算损失
            loss = criterion(batch_pred_HIC, batch_y_HIC)
            loss_batch.append(loss.item())

            # 如果使用了 y_transform，需要将HIC 值反变换回原始范围以计算指标
            if y_transform is not None:
                batch_pred_HIC = y_transform.inverse(batch_pred_HIC)
                batch_y_HIC = y_transform.inverse(batch_y_HIC)

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
        print("Warning: HIC_preds or HIC_trues contains inf values. Replacing inf with large finite values.")
        HIC_preds = np.nan_to_num(HIC_preds, nan=0.0, posinf=1e6, neginf=-1e6)
        HIC_trues = np.nan_to_num(HIC_trues, nan=0.0, posinf=1e6, neginf=-1e6)

    # 计算 AIS 预测值
    AIS_preds = AIS_cal(HIC_preds)

    # 计算准确率和 MAE
    accuracy = 100. * (1 - np.count_nonzero(AIS_preds - AIS_trues) / len(AIS_trues))
    mae = mean_absolute_error(HIC_trues, HIC_preds)

    return avg_loss, accuracy, mae


if __name__ == "__main__":
    ''' Train the TCN-based post-crash injury prediction model, i.e., the teacher model. '''
    from torch.utils.tensorboard import SummaryWriter

    # 创建独立文件夹保存本次运行结果
    current_time = datetime.now().strftime("%m%d%H%M")  # 月日时分
    run_dir = os.path.join("./runs", f"TeacherModel_Train_{current_time}")
    os.makedirs(run_dir, exist_ok=True)

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=run_dir)

    # 定义优化相关的超参数
    Epochs = 500
    Batch_size = 64
    Learning_rate = 0.005 # 初始学习率
    Learning_rate_min = 1e-6 # 余弦退火最小学习率
    weight_decay = 0.0001 # L2 正则化系数
    Patience = 5 # 早停等待轮数

    # 定义模型相关的超参数
    num_blocks_of_tcn = 4  # TCN 的块数
    num_layers_of_mlpE = 4  # MLP 编码器的层数
    num_layers_of_mlpD = 4  # MLP 解码器的层数
    mlpE_hidden = 128  # MLP 编码器的隐藏层维度
    mlpD_hidden = 96  # MLP 解码器的隐藏层维度
    encoder_output_dim = 128  # 编码器输出特征维度
    decoder_output_dim = 16  # 解码器输出特征维度
    dropout = 0.1 # Dropout 概率

    # 设定数据集大小
    train_size = 5000
    val_size = 500

    # 加载标签变换对象和数据集
    HIC_transform = SigmoidTransform(lower_bound=0, upper_bound=2500)
    dataset = CrashDataset(y_transform=HIC_transform)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])
    torch.save(train_dataset, "./data/train_dataset.pt"), torch.save(val_dataset, "./data/val_dataset.pt"), torch.save(test_dataset, "./data/test_dataset.pt")

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

    # RTX 4060Ti
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型、优化器和损失函数
    model = models.TeacherModel(
        num_classes_of_discrete=dataset.num_classes_of_discrete,
        num_blocks_of_tcn=num_blocks_of_tcn,
        num_layers_of_mlpE=num_layers_of_mlpE,
        num_layers_of_mlpD=num_layers_of_mlpD,
        mlpE_hidden=mlpE_hidden,
        mlpD_hidden=mlpD_hidden,
        encoder_output_dim=encoder_output_dim,
        decoder_output_dim=decoder_output_dim,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)

    # 初始化变量
    LossCurve_val, LossCurve_train = [], []
    Best_accu = 0
    Best_mae = 1e6

    # 主训练循环
    for epoch in range(Epochs):
        epoch_start_time = time.time()

        # 训练模型
        train_loss = train(model, train_loader, optimizer, criterion, device)
        LossCurve_train.append(train_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Train Loss: {train_loss:.3f}")

        # 验证模型
        val_loss, val_accuracy, val_mae = valid(model, val_loader, criterion, device, y_transform=dataset.y_transform)
        LossCurve_val.append(val_loss)
        print(f"            | Val Loss: {val_loss:.3f} | Val Accuracy: {val_accuracy:.1f}% | MAE: {val_mae:.1f}")

        # 学习率调整
        scheduler.step()

        # TensorBoard 记录
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_accuracy, epoch)
        writer.add_scalar("Mae/Val", val_mae, epoch)

        # 保存分类准确率最佳的模型
        if val_accuracy > Best_accu:
            Best_accu = val_accuracy
            if Best_accu > 80:
                torch.save(model.state_dict(), os.path.join(run_dir, "teacher_best_accu.pth"))
                print(f"Best model saved with val accuracy: {Best_accu:.2f}%")

        # 保存 MAE 最佳的模型
        if val_mae < Best_mae:
            Best_mae = val_mae
            if Best_mae < 500:
                torch.save(model.state_dict(), os.path.join(run_dir, "teacher_best_mae.pth"))
                print(f"Best model saved with val MAE: {Best_mae:.1f}")

        # 早停逻辑
        if len(LossCurve_val) > Patience:
            recent_losses = LossCurve_val[-Patience:]
            if all(recent_losses[i] < recent_losses[i + 1] for i in range(len(recent_losses) - 1)): # 如果最近连续 Patience 轮 val loss 都上升
                print(f"Early Stop at epoch: {epoch+1}! Last val accuracy: {val_accuracy:.1f}%! Best val accuracy: {Best_accu:.1f}%")
                print(f"Last val MAE: {val_mae:.1f}! Best val MAE: {Best_mae:.1f}")
                break

        print(f"            | Time: {time.time()-epoch_start_time:.2f}s")

    # 关闭 TensorBoard
    writer.close()

    # 保存超参数和关键结果到 JSON 文件
    results = {
        "hyperparameters related to training": {
            "Epochs": Epochs,
            "Batch_size": Batch_size,
            "Learning_rate": float(Learning_rate),  # 转换为 float
            "Learning_rate_min": float(Learning_rate_min),  # 转换为 float
            "weight_decay": float(weight_decay),  # 转换为 float
            "Patience": Patience,
        },
        "hyperparameters related to model": {
            "num_blocks_of_tcn": num_blocks_of_tcn,
            "num_layers_of_mlpE": num_layers_of_mlpE,
            "num_layers_of_mlpD": num_layers_of_mlpD,
            "mlpE_hidden": mlpE_hidden,
            "mlpD_hidden": mlpD_hidden,
            "encoder_output_dim": encoder_output_dim,
            "decoder_output_dim": decoder_output_dim,
            "dropout": float(dropout),  # 转换为 float
        },
        "results": {
            "final_epoch": epoch + 1,
            "Best_accu": float(Best_accu),  # 转换为 float
            "Best_mae": float(Best_mae),  # 转换为 float
            "last_val_accuracy": float(val_accuracy),  # 转换为 float
            "last_val_mae": float(val_mae),  # 转换为 float
        }
    }
    if dataset.y_transform is not None:
        results["hyperparameters related to training"]["HIC_transform"] = {
            "lower_bound": dataset.y_transform.lower_bound,
            "upper_bound": dataset.y_transform.upper_bound,
        }
        
    with open(os.path.join(run_dir, "TrainingRecord.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Training Finished!")