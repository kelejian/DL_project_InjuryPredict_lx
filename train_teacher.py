# -*- coding: utf-8 -*-
# 2024.12.19
import os
import time
import torch
import random
import warnings
import numpy as np
from torch import nn
from torch.utils.data import  DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, root_mean_squared_error

from utils import models, dataset_prepare, combined_loss

import wandb

warnings.filterwarnings('ignore')

# Define the random seed.
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train(model, loader, optimizer, criterion):
    model.train()
    loss_batch = []
    for batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS in loader:
        batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS = batch_x_acc.to(device), batch_x_att.to(device), batch_y_HIC.to(device), batch_y_AIS.to(device)
        batch_pred_HIC, _, _ = model(batch_x_acc, batch_x_att[:, 5:]) # x_att[:, 5:] 为teacher model的额外特征数据
        loss = criterion(batch_pred_HIC, batch_y_HIC)
        #loss, _, _ = criterion(batch_pred_HIC, batch_y_HIC, batch_y_AIS)
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        loss_batch.append(loss.item())
    return np.mean(loss_batch)

def valid(model, loader, criterion):
    model.eval()
    loss_batch = []
    all_HIC_preds = []
    all_HIC_trues = []
    all_AIS_trues = []
    
    with torch.no_grad():
        for batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS in loader:
            batch_x_acc, batch_x_att, batch_y_HIC = batch_x_acc.to(device), batch_x_att.to(device), batch_y_HIC.to(device)
            batch_pred_HIC, _, _ = model(batch_x_acc, batch_x_att[:, 5:]) # x_att[:, 5:] 为teacher model的额外特征数据
            loss = criterion(batch_pred_HIC, batch_y_HIC)
            #loss, _, _ = criterion(batch_pred_HIC, batch_y_HIC, batch_y_AIS)
            loss_batch.append(loss.item())
            all_HIC_preds.append(batch_pred_HIC.cpu().numpy())
            all_HIC_trues.append(batch_y_HIC.cpu().numpy())
            all_AIS_trues.append(batch_y_AIS.numpy())

    avg_loss = np.mean(loss_batch)
    HIC_preds = np.concatenate(all_HIC_preds)
    HIC_trues = np.concatenate(all_HIC_trues)
    AIS_preds = dataset_prepare.AIS_cal(HIC_preds)
    AIS_trues = np.concatenate(all_AIS_trues)
    accuracy = 100. * (1 - np.count_nonzero(AIS_preds - AIS_trues) / len(AIS_trues))
    rmse = root_mean_squared_error(HIC_trues, HIC_preds)
    # conf_mat = confusion_matrix(AIS_trues, AIS_preds)
    # G_mean = geometric_mean_score(AIS_trues, AIS_preds)
    # report = classification_report_imbalanced(AIS_trues, AIS_preds, digits=3)

    return avg_loss, accuracy, rmse


if __name__ == "__main__":
    ''' Train the TCN-based post-crash injury prediction model, i.e., the teacher model. '''

    # Initialize #wandb.
    #wandb.init(project="Injury_predic_DL", name="Train_teacher_model")

    # Define hyper-parameters related to optimization.
    Epochs = 500
    Batch_size = 64
    Learning_rate = 0.005
    Learning_rate_min = 1e-6
    weight_decay = 0
    Patience = 5
    #mse_overweight = 1.5
    # Define hyper-parameters related to the model.
    Level_Size = 5
    K_size = 5
    Emb_size = 128
    Hidden_size = 64
    Dropout = 0.3
    Num_Chans = [Hidden_size] * (Level_Size - 1) + [Emb_size]

    # Define the dataset size.
    train_size = 5000
    val_size = 500
    #  Update the hyper-parameters into wandb.
    #wandb.config.update({
    #     "Epochs": Epochs,
    #     "Batch_size": Batch_size,
    #     "Learning_rate": Learning_rate,
    #     "Learning_rate_min": Learning_rate_min,
    #     "weight_decay": weight_decay,
    #     "Patience": Patience,
    #     "Level_Size": Level_Size,
    #     "K_size": K_size,
    #     "Emb_size": Emb_size,
    #     "Hidden_size": Hidden_size,
    #     "Dropout": Dropout,
    #     "Num_Chans": Num_Chans,
    #     "mse_overweight": mse_overweight
    # })

    # Load the dataset.
    dataset = dataset_prepare.CrashDataset()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])
    torch.save(train_dataset, "./data/train_dataset.pt"), torch.save(val_dataset, "./data/val_dataset.pt"), torch.save(test_dataset, "./data/test_dataset.pt")

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0) 

    # RTX 4060Ti
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model, optimizer, and criterion.
    model = models.teacher_model(Emb_size, Num_Chans, K_size, dropout=Dropout, emb_dropout=Dropout).to(device)
    criterion = nn.MSELoss().to(device)
    #MSEloss_expect, Classification_loss_expect = dataset.Loss_expect_cal()
    #criterion = combined_loss.CombinedLoss(mse_overweight=mse_overweight, mse_norm_coef = MSEloss_expect, class_norm_coef=Classification_loss_expect).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min) # T_max=Epochs: 学习率在整个训练周期内从大到小按余弦衰减

    LossCurve_val, LossCurve_train = [], []
    Best_accu = 0

    # 主训练循环
    #torch.autograd.set_detect_anomaly(True) # 检测梯度异常,会降低训练速度
    for epoch in range(Epochs):
        epoch_start_time = time.time()

        # 训练模型
        train_loss = train(model, train_loader, optimizer, criterion)
        LossCurve_train.append(train_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Train Loss: {train_loss:.3f}")

        # 验证模型
        val_loss, val_accuracy, val_rmse = valid(model, val_loader, criterion)
        LossCurve_val.append(val_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_accuracy:.1f}% | RMSE: {val_rmse:.1f}")

        # 学习率调整
        scheduler.step()

        # #wandb 记录
        #wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy, "RMSE": rmse})

        # 保存最佳模型
        if val_accuracy > Best_accu:
            Best_accu = val_accuracy
            if Best_accu > 80:
                torch.save(model.state_dict(), "./ckpt/teacher_best.pth")
                #wandb.save("teacher_best.pth")
                print(f"Best model saved with val accuracy: {Best_accu:.1f}%")
        # 早停逻辑
        if len(LossCurve_val) > Patience:
            recent_losses = LossCurve_val[-Patience:]
            if all(recent_losses[i] < recent_losses[i + 1] for i in range(len(recent_losses) - 1)):
                print(f"Early Stop at epoch: {epoch+1}! Last val accuracy: {val_accuracy:.1f}%! Best val accuracy: {Best_accu:.1f}%")
                break

        print(f"Epoch {epoch+1}/{Epochs} | Time: {time.time()-epoch_start_time:.2f}s")

    #wandb.finish()
    print("Training Finished!")