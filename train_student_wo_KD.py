import os
import time
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from utils import models, dataset_prepare
import wandb
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train(model, loader, optimizer, criterion, device):
    model.train()
    loss_batch = []

    for batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS in loader:
        batch_x_acc = batch_x_acc.to(device)
        batch_x_att = batch_x_att.to(device)
        batch_y_HIC = batch_y_HIC.to(device)

        # Forward pass
        pred_HIC, _, _ = model(batch_x_att)
        loss = criterion(pred_HIC, batch_y_HIC)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch.append(loss.item())

    return np.mean(loss_batch)

def valid(model, loader, criterion, device):
    model.eval()
    loss_batch = []
    all_HIC_preds = []
    all_HIC_trues = []
    all_AIS_trues = []

    with torch.no_grad():
        for batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS in loader:
            batch_x_acc = batch_x_acc.to(device)
            batch_x_att = batch_x_att.to(device)
            batch_y_HIC = batch_y_HIC.to(device)

            # Forward pass
            pred_HIC, _, _ = model(batch_x_att)

            # Compute loss
            loss = criterion(pred_HIC, batch_y_HIC)
            loss_batch.append(loss.item())

            # Collect predictions and true values
            all_HIC_preds.append(pred_HIC.cpu().numpy())
            all_HIC_trues.append(batch_y_HIC.cpu().numpy())
            all_AIS_trues.append(batch_y_AIS.numpy())

    # Combine and compute metrics
    avg_loss = np.mean(loss_batch)
    HIC_preds = np.concatenate(all_HIC_preds)
    HIC_trues = np.concatenate(all_HIC_trues)
    AIS_preds = dataset_prepare.AIS_cal(HIC_preds)
    AIS_trues = np.concatenate(all_AIS_trues)
    accuracy = 100. * (1 - np.count_nonzero(AIS_preds - AIS_trues) / len(AIS_trues))
    rmse = root_mean_squared_error(HIC_trues, HIC_preds)

    return avg_loss, accuracy, rmse

if __name__ == "__main__":
    # 初始化 wandb
    wandb.init(project="Injury_predic_DL", name="Train_student_wo_KD")

    # 超参数设置
    Epochs = 500
    Batch_size = 64
    Learning_rate = 0.005
    Learning_rate_min = 1e-6
    Patience = 10

    Emb_size = 128
    # Level_Size = 5
    # K_size = 5
    # Hidden_size = 64
    # Dropout = 0.3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集加载
    dataset = dataset_prepare.CrashDataset()
    train_size = 5000
    val_size = 500
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, len(dataset) - train_size - val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

    # 加载模型
    model = models.student_model(Emb_size).to(device)

    # 定义优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)

    # 定义损失函数
    criterion = nn.MSELoss().to(device)

    # 训练循环
    Best_accu = 0
    LossCurve_train, LossCurve_val = [], []

    for epoch in range(Epochs):
        start_time = time.time()

        # 训练
        train_loss = train(model, train_loader, optimizer, criterion, device)
        LossCurve_train.append(train_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Train Loss: {train_loss:.3f}")

        # 验证
        val_loss, val_accuracy, val_rmse = valid(model, val_loader, criterion, device)
        LossCurve_val.append(val_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_accuracy:.1f}% | RMSE: {val_rmse:.1f}")

        # 学习率调整
        scheduler.step()

        # wandb 日志记录
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy, "val_rmse": val_rmse})

        # 保存最佳模型
        # 保存最佳模型
        if val_accuracy > Best_accu:
            Best_accu = val_accuracy
            if Best_accu > 72:
                torch.save(model.state_dict(), './ckpt/student_wo_KD_best.pth')
                if Best_accu > 75:
                    wandb.save("student_wo_KD_best.pth")
                print(f"Best model saved with val accuracy: {Best_accu:.1f}%")

        # 早停逻辑
        if len(LossCurve_val) > Patience:
            recent_losses = LossCurve_val[-Patience:]
            if all(recent_losses[i] < recent_losses[i + 1] for i in range(len(recent_losses) - 1)):
                print(f"Early Stop at epoch: {epoch + 1}! Best Val Accuracy: {Best_accu:.2f}%")
                break

    wandb.finish()
    print("Training Finished!! Best Val accuracy: {Best_accu:.1f}%")
