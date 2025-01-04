import os
import time
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from utils import models, dataset_prepare
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, root_mean_squared_error
import warnings
import wandb

warnings.filterwarnings('ignore')

# 设置随机种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train(student_model, teacher_model, loader, optimizer, criterion, ratio_E, ratio_D, device):
    student_model.train()
    teacher_model.eval()
    loss_batch = []

    for batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS in loader:
        batch_x_acc = batch_x_acc.to(device)
        batch_x_att = batch_x_att.to(device)
        batch_y_HIC = batch_y_HIC.to(device)

        # Student 和 Teacher 模型的前向传播
        pred_HIC_s, pred_D_s, pred_E_s = student_model(batch_x_att) # (batch_size,), (batch_size, 16), (batch_size, num_channels[-1]=128)
        with torch.no_grad():
            pred_HIC_t, pred_D_t, pred_E_t = teacher_model(batch_x_acc, batch_x_att[:, 5:]) # (batch_size,), (batch_size, 16), (batch_size, num_channels[-1]=128)

        # 蒸馏损失和预测损失
        loss_pred = criterion(pred_HIC_s, batch_y_HIC)
        loss_KD_E = criterion(pred_E_s, pred_E_t)
        loss_KD_D = criterion(pred_D_s, pred_D_t)

        loss = loss_pred + ratio_E * loss_KD_E + ratio_D * loss_KD_D

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch.append(loss.item())

    return np.mean(loss_batch)

def valid(student_model, teacher_model, loader, criterion, ratio_E, ratio_D, device):
    student_model.eval()
    teacher_model.eval()
    loss_batch = []
    all_HIC_preds = []
    all_HIC_trues = []
    all_AIS_trues = []

    with torch.no_grad():
        for batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS in loader:
            batch_x_acc, batch_x_att, batch_y_HIC = batch_x_acc.to(device), batch_x_att.to(device), batch_y_HIC.to(device)

            pred_HIC_s, pred_D_s, pred_E_s = student_model(batch_x_att)
            pred_HIC_t, pred_D_t, pred_E_t = teacher_model(batch_x_acc, batch_x_att[:, 5:])

            loss_pred = criterion(pred_HIC_s, batch_y_HIC)
            loss_KD_E = criterion(pred_E_s, pred_E_t)
            loss_KD_D = criterion(pred_D_s, pred_D_t)

            loss = loss_pred + ratio_E * loss_KD_E + ratio_D * loss_KD_D

            loss_batch.append(loss.item())

            all_HIC_preds.append(pred_HIC_s.cpu().numpy())
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

    # Initialize #wandb.
    wandb.init(project="Injury_predic_DL", name="Train_student_model_W_KD")

    # 训练超参
    Epochs = 500
    Batch_size = 128
    Learning_rate = 0.005
    Learning_rate_min = 1e-6
    patience = 10  # 早停的耐心值
    ratio_E = 20000
    ratio_D = 4000

    # 模型超参
    Emb_size_teacher = 128  # 教师模型的 embedding size
    Emb_size = 128 # 学生模型的 embedding size需要和教师模型的Num_Chans_teacher[-1] = Emb_size_teacher一致
    Level_Size = 5
    K_size = 5
    Hidden_size = 64
    Dropout = 0.3
    Num_Chans_teacher = [Hidden_size] * (Level_Size - 1) + [Emb_size_teacher]
    Num_Chans_student = [Hidden_size] * (Level_Size - 1) + [Emb_size]

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

    # 加载 Teacher 模型
    teacher_model = models.teacher_model(Emb_size_teacher, Num_Chans_teacher, kernel_size=K_size, dropout=Dropout).to(device)
    teacher_weights_path = './ckpt/teacher_best.pth'
    if os.path.exists(teacher_weights_path):
        teacher_model.load_state_dict(torch.load(teacher_weights_path))
        teacher_model.eval()
        print("Teacher model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Teacher model weights not found at {teacher_weights_path}.")

    # 加载 Student 模型
    student_model = models.student_model(Emb_size).to(device)
    if Emb_size == Emb_size_teacher:
        pretrained_dict = torch.load(teacher_weights_path)
        model_dict = student_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        student_model.load_state_dict(model_dict)
        print("Pretrained weights partially loaded into Student model.")
    else:
        print("Embedding sizes do not match between Teacher and Student models. No weights transferred.")

    # 加载优化器和学习率调度器
    optimizer = optim.AdamW(student_model.parameters(), lr=Learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)

    # 定义损失函数
    criterion = nn.MSELoss().to(device)
    
    Best_accu = 0
    LossCurve_val, LossCurve_train = [], []

    # 训练循环
    for epoch in range(Epochs):
        epoch_start_time = time.time()

        # 训练
        train_loss = train(student_model, teacher_model, train_loader, optimizer, criterion, ratio_E, ratio_D, device)
        LossCurve_train.append(train_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Train Loss: {train_loss:.3f}")

        # 验证
        val_loss, val_accuracy, val_rmse =  valid(student_model, teacher_model, val_loader, criterion, ratio_E, ratio_D, device)
        LossCurve_val.append(val_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_accuracy:.1f}% | RMSE: {val_rmse:.1f}")

        # 学习率调整
        scheduler.step()

        # wandb 记录
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy, "RMSE": val_rmse})

        # 保存最佳模型
        if val_accuracy > Best_accu:
            Best_accu = val_accuracy
            if Best_accu > 80:
                torch.save(student_model.state_dict(), './ckpt/student_w_KD_best.pth')
                wandb.save("student_w_KD_best.pth")
                print(f"Best model saved with val accuracy: {Best_accu:.1f}%")

        # 早停逻辑
        if len(LossCurve_val) > patience:
            recent_losses = LossCurve_val[-patience:]
            if all(recent_losses[i] < recent_losses[i + 1] for i in range(len(recent_losses) - 1)):
                print(f"Early Stop at epoch: {epoch + 1}! Best Val accuracy: {Best_accu:.1f}%")
                break

        print(f"Epoch {epoch+1}/{Epochs} | Time: {time.time()-epoch_start_time:.2f}s")

    wandb.finish()
    print(f"Training Finished!! Best Val accuracy: {Best_accu:.1f}%")
