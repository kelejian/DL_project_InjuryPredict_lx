# -*- coding: utf-8 -*-
# 2025.1.7
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

#import wandb

warnings.filterwarnings('ignore')

# Define the random seed.
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train_distill(model, teacher_model, loader, optimizer, criterion, device, distill_encoder_weight, distill_decoder_weight):
    """
    使用知识蒸馏训练学生模型。

    参数:
        model: 学生模型实例。
        teacher_model: 教师模型实例。
        loader: 数据加载器。
        optimizer: 优化器。
        criterion: 损失函数。
        device: GPU 或 CPU。
        distill_encoder_weight: 编码器蒸馏损失的权重。
        distill_decoder_weight: 解码器蒸馏损失的权重。
    """
    model.train()
    teacher_model.eval()  # 教师模型固定，不更新参数
    loss_batch = []
    distill_encoder_loss_batch = []
    distill_decoder_loss_batch = []

    for batch_x_acc, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, _ in loader:
        # 将数据移动到设备
        batch_x_acc = batch_x_acc.to(device)
        batch_x_att_continuous = batch_x_att_continuous.to(device)
        batch_x_att_discrete = batch_x_att_discrete.to(device)
        batch_y_HIC = batch_y_HIC.to(device)

        # 学生模型前向传播
        student_hic_pred, student_encoder_output, student_decoder_output = model(batch_x_att_continuous, batch_x_att_discrete)

        # 教师模型前向传播
        with torch.no_grad():
            _, teacher_encoder_output, teacher_decoder_output = teacher_model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)

        # 计算回归损失
        regression_loss = criterion(student_hic_pred, batch_y_HIC)

        # 计算蒸馏损失
        distill_encoder_loss = nn.MSELoss()(student_encoder_output, teacher_encoder_output)
        distill_decoder_loss = nn.MSELoss()(student_decoder_output, teacher_decoder_output)

        # 总损失
        total_loss = (
            regression_loss
            + distill_encoder_weight * distill_encoder_loss
            + distill_decoder_weight * distill_decoder_loss
        )

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 记录损失
        loss_batch.append(total_loss.item())
        distill_encoder_loss_batch.append(distill_encoder_loss.item())
        distill_decoder_loss_batch.append(distill_decoder_loss.item())

    return (
        np.mean(loss_batch),
        np.mean(distill_encoder_loss_batch),
        np.mean(distill_decoder_loss_batch),
    )

def valid_distill(model, loader, criterion, device, y_transform=None):
    """
    学生模型的验证函数。

    参数:
        model: 学生模型实例。
        loader: 数据加载器。
        criterion: 损失函数。
        device: GPU 或 CPU。
        y_transform: 数据集中的HIC标签变换对象。若数据集中的HIC标签没有进行变换则为None。
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
    ''' Train the student model with knowledge distillation. '''
    from torch.utils.tensorboard import SummaryWriter

    # 创建独立文件夹保存本次运行结果
    current_time = datetime.now().strftime("%m%d%H%M")  # 月日时分
    run_dir = os.path.join("./runs", f"StudentModel_Distill_{current_time}")
    os.makedirs(run_dir, exist_ok=True)

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=run_dir)

    # 加载教师模型
    teacher_run_dir = ".\\runs\\TeacherModel_Train_01121600"
    teacher_model_name = "teacher_best_mae.pth"  # 或 teacher_best_accu.pth
    teacher_model_path = os.path.join(teacher_run_dir, teacher_model_name)

    # 从教师模型的 JSON 文件中读取超参数
    with open(os.path.join(teacher_run_dir, "TrainingRecord.json"), "r") as f:
        teacher_hyperparams = json.load(f)["hyperparameters related to model"]

    ############################################################################################
    ############################################################################################
    # 学生模型超参数
    # 定义优化相关的超参数
    Epochs = 500
    Batch_size = 512
    Learning_rate = 0.01  # 初始学习率
    Learning_rate_min = 5e-7  # 余弦退火最小学习率
    weight_decay = 0.0001  # L2 正则化系数
    Patience = 8  # 早停等待轮数
    base_loss = "mae"  # 或 "mae"
    weight_factor_classify = 1.7  # 加权损失函数的系数1
    weight_factor_sample = 0.5  # 加权损失函数的系数2
    distill_encoder_weight = 100.0  # 编码器蒸馏损失的权重; 和mae loss差了250-300倍
    distill_decoder_weight = 30.0  # 解码器蒸馏损失的权重；和mae loss差了800-900倍
    # 定义模型相关的超参数
    num_layers_of_mlpE = 4  # MLP 编码器的层数
    num_layers_of_mlpD = 3  # MLP 解码器的层数
    mlpE_hidden = 128  # MLP 编码器的隐藏层维度
    mlpD_hidden = 128  # MLP 解码器的隐藏层维度
    encoder_output_dim = teacher_hyperparams["encoder_output_dim"]  # 编码器输出特征维度
    decoder_output_dim = teacher_hyperparams["decoder_output_dim"]  # 解码器输出特征维度
    dropout = 0.05  # Dropout 概率
    # 是否使用 HIC 标签变换对象
    lower_bound = 0  # HIC 标签的下界
    upper_bound = 2500  # HIC 标签的上界
    HIC_transform = None  # HIC 标签变换对象 或 SigmoidTransform(lower_bound, upper_bound) 暂不用
    ############################################################################################
    ############################################################################################

    # 加载数据集对象
    dataset = CrashDataset(y_transform=HIC_transform)
    # 从data文件夹直接加载数据集
    if dataset.y_transform is None:
        train_dataset = torch.load("./data/train_dataset.pt")
        val_dataset = torch.load("./data/val_dataset.pt")
    else:
        train_dataset = torch.load("./data/train_dataset_ytrans.pt")
        val_dataset = torch.load("./data/val_dataset_ytrans.pt")

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

    # 初始化教师模型
    teacher_model = models.TeacherModel(
        Ksize_init=teacher_hyperparams["Ksize_init"],
        Ksize_mid=teacher_hyperparams["Ksize_mid"],
        num_classes_of_discrete=dataset.num_classes_of_discrete,
        num_blocks_of_tcn=teacher_hyperparams["num_blocks_of_tcn"],
        num_layers_of_mlpE=teacher_hyperparams["num_layers_of_mlpE"],
        num_layers_of_mlpD=teacher_hyperparams["num_layers_of_mlpD"],
        mlpE_hidden=teacher_hyperparams["mlpE_hidden"],
        mlpD_hidden=teacher_hyperparams["mlpD_hidden"],
        encoder_output_dim=teacher_hyperparams["encoder_output_dim"],
        decoder_output_dim=teacher_hyperparams["decoder_output_dim"],
        dropout=teacher_hyperparams["dropout"]
    ).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path))
    teacher_model.eval()  # 教师模型固定，不更新参数

    # 定义损失函数、优化器和学习率调度器
    criterion = weighted_loss(base_loss, weight_factor_classify, weight_factor_sample, dataset.y_transform)
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
        train_loss, distill_encoder_loss, distill_decoder_loss = train_distill(
            model, teacher_model, train_loader, optimizer, criterion, device, distill_encoder_weight, distill_decoder_weight
        )
        LossCurve_train.append(train_loss)
        print(f"Epoch {epoch+1}/{Epochs} | Train Loss: {train_loss:.3f} | Encoder Distill Loss: {distill_encoder_loss:.3f} | Decoder Distill Loss: {distill_decoder_loss:.3f}")

        # 验证模型
        val_loss, val_accuracy, val_mae, val_rmse = valid_distill(model, val_loader, criterion, device, y_transform=dataset.y_transform)
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
            if Best_accu > 80:
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
            "distill_encoder_weight": distill_encoder_weight,
            "distill_decoder_weight": distill_decoder_weight,
            "teacher_run_dir": teacher_run_dir,
            "teacher_model_name": teacher_model_name
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

    if dataset.y_transform is not None:
        results["hyperparameters related to training"]["HIC_transform"] = {
            "lower_bound": dataset.y_transform.lower_bound,
            "upper_bound": dataset.y_transform.upper_bound,
        }

    with open(os.path.join(run_dir, "TrainingRecord.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Training Finished!")