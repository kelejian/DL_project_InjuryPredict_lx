import optuna
import torch, os, json
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import random
from utils import models
from utils.dataset_prepare import CrashDataset
from utils.AIS_cal import AIS_3_cal_head, AIS_cal_head, AIS_cal_chest, AIS_cal_neck 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from utils.weighted_loss import weighted_loss
from optuna.trial import TrialState
import joblib
from optuna import TrialPruned
from optuna.storages import RDBStorage

from utils.set_random_seed import set_random_seed
set_random_seed()

def train_distill(model, teacher_model, loader, optimizer, criterion, device, distill_encoder_weight, distill_decoder_weight):
    """
    使用知识蒸馏训练学生模型。
    """
    model.train()
    teacher_model.eval()  # 教师模型固定，不更新参数
    for batch_x_acc, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, _ in loader:
        # 将数据移动到设备
        batch_x_acc = batch_x_acc.to(device)  # 确保 batch_x_acc 也在 GPU 上
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

def valid_distill(model, loader, criterion, device):
    """
    学生模型的验证函数。
    """
    model.eval()
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

            # 记录预测值和真实值
            all_HIC_preds.append(batch_pred_HIC.cpu().numpy())
            all_HIC_trues.append(batch_y_HIC.cpu().numpy())
            all_AIS_trues.append(batch_y_AIS.cpu().numpy())

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
    AIS_preds = AIS_cal_head(HIC_preds)
    accuracy = 100. * (1 - np.count_nonzero(AIS_preds - AIS_trues) / len(AIS_trues))

    # 计算MAE, RMSE
    HIC_preds[HIC_preds > 2500] = 2500 # 规定上界，过高的HIC值(>2000基本就危重伤)不具有实际意义
    HIC_trues[HIC_trues > 2500] = 2500  
    mae = mean_absolute_error(HIC_trues, HIC_preds)
    rmse = root_mean_squared_error(HIC_trues, HIC_preds)

    return accuracy, mae, rmse

def objective(trial):
    # 超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 0.008, 0.015, step=0.001)  # 学习率
    #batch_size = trial.suggest_categorical("batch_size", [256, 512])  # 批量大小
    batch_size = 512
    #weight_decay = trial.suggest_float("weight_decay", 5e-5, 5e-4, step=5e-5)  # 权重衰减
    weight_decay = 2e-4
    # 蒸馏损失权重
    distill_encoder_weight = trial.suggest_float("distill_encoder_weight", 0, 200, step=10.0) # 编码器蒸馏损失的权重; 和mae loss差了250-300倍
    distill_decoder_weight = trial.suggest_float("distill_decoder_weight", 0, 100, step=10.0) # 解码器蒸馏损失的权重; 和mae loss差了800-900倍

    # 学生模型结构超参数
    num_layers_of_mlpE = trial.suggest_int("num_layers_of_mlpE", 2, 4, step=1)  # MLP 编码器层数
    #num_layers_of_mlpD = trial.suggest_int("num_layers_of_mlpD", 3, 5, step=1)  # MLP 解码器层数
    num_layers_of_mlpD = 3
    mlpE_hidden = trial.suggest_int("mlpE_hidden", 128, 256, step=64)  # MLP 编码器隐藏层维度
    mlpD_hidden = trial.suggest_int("mlpD_hidden", 128, 256, step=64)  # MLP 解码器隐藏层维度
    dropout = trial.suggest_float("dropout", 0.05, 0.1, step=0.05)                         # Dropout 概率
    #base_loss = trial.suggest_categorical("base_loss", ["mse", "mae"])         # 基础损失函数
    base_loss = "mae"
    weight_factor_classify = trial.suggest_float("weight_factor_classify", 1.6, 2.0, step=0.1)
    weight_factor_sample =trial.suggest_float("weight_factor_sample", 0.3, 0.7, step=0.1)
    Epochs = 500
    eta_min = 5e-6

    # 加载数据集对象
    dataset = CrashDataset()
    # 从data文件夹直接加载数据集

    train_dataset = torch.load("./data/train_dataset.pt")
    val_dataset = torch.load("./data/val_dataset.pt")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载教师模型
    teacher_run_dir = ".\\runs\\TeacherModel_Train_01121600"
    teacher_model_name = "teacher_best_mae.pth"
    teacher_model_path = os.path.join(teacher_run_dir, teacher_model_name)

    with open(os.path.join(teacher_run_dir, "TrainingRecord.json"), "r") as f:
        teacher_hyperparams = json.load(f)["hyperparameters related to model"]

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
    teacher_model.eval()

    # 构建学生模型
    model = models.StudentModel(
        num_classes_of_discrete=dataset.num_classes_of_discrete,
        num_layers_of_mlpE=num_layers_of_mlpE,
        num_layers_of_mlpD=num_layers_of_mlpD,
        mlpE_hidden=mlpE_hidden,
        mlpD_hidden=mlpD_hidden,
        encoder_output_dim=teacher_hyperparams["encoder_output_dim"],
        decoder_output_dim=teacher_hyperparams["decoder_output_dim"],
        dropout=dropout
    ).to(device)

    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = weighted_loss(base_loss=base_loss, weight_factor_classify=weight_factor_classify, weight_factor_sample=weight_factor_sample).to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=eta_min)

    val_maes = []
    val_accuracies = []
    val_rmses = []
    # 训练模型
    for epoch in range(Epochs//10*3):

        train_distill(model, teacher_model, train_loader, optimizer, criterion, device, distill_encoder_weight, distill_decoder_weight)
        
        val_accuracy, val_mae, val_rmse = valid_distill(model, val_loader, criterion, device)

        # prune_score = (val_mae * val_rmse) ** 0.5 
        # trial.report(prune_score , epoch) 
        # if trial.should_prune():  # 检查是否应该剪枝
        #     raise optuna.TrialPruned()  # 剪枝当前 trial

        # 记录验证集 MAE
        if epoch > Epochs//10*3 - 10:
            val_accuracies.append(val_accuracy)
            val_maes.append(val_mae)
            val_rmses.append(val_rmse)

        # 更新学习率
        scheduler.step()

    return np.mean(val_accuracies), np.mean(val_maes), np.mean(val_rmses)

# 定义保存和加载的文件路径
study_file = "./runs/optuna_study_student2.pkl"
study_name = "student_model_optimization2"
db_path = "sqlite:///./runs/optuna_study.db"

# 定义 SQLite 数据库路径
storage = RDBStorage(db_path)

# 创建中位数剪枝器
# pruner = optuna.pruners.MedianPruner(
#     n_startup_trials=20,  # 前几次试验不剪枝
#     n_warmup_steps=50,     # 每个试验的前几步不剪枝
#     interval_steps=5,     # 每隔几步评估一次剪枝条件
#     n_min_trials=20        # 计算中位数所需的最小历史试验数量
# )

# 检查是否存在已有的研究
try:
    study = optuna.load_study(study_name=study_name, storage=storage)
except Exception as e:
    print(f"Failed to load study: {str(e)}")
    print("Creating new study...")
    study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler(), study_name=study_name, storage=storage, directions=["maximize", "minimize", "minimize"]) # 多目标优化使用NSGAIISampler


# 定义回调函数，定期保存研究结果
def save_study_callback(study, trial, freq=5):
    if trial.number % freq == 0:  
        pareto_front = study.best_trials
        joblib.dump(pareto_front, study_file)
        print('=' * 10)
        print(f"Study saved at trial {trial.number}")
        print('=' * 10)

# 运行优化
try:
    study.optimize(objective, n_trials=100, callbacks=[save_study_callback])
except KeyboardInterrupt:
    print("Optimization interrupted by user.")
    joblib.dump(study, study_file)
except Exception as e:
    print(f"Optimization interrupted due to error: {str(e)}")
    joblib.dump(study, study_file)

# 输出 Pareto 前沿
print("=" * 50)
print("Pareto Front:")
for trial in study.best_trials:
    print(f"Accuracy: {trial.values[0]:.2f}%, MAE: {trial.values[1]:.2f}, RMSE: {trial.values[2]:.2f}")
    print(f"Hyperparameters: {trial.params}")
    print("-" * 50)
print("=" * 50)