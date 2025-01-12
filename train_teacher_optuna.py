import optuna
import torch, os, json
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import random
from utils import models
from utils.dataset_prepare import CrashDataset, AIS_cal
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from utils.weighted_loss import weighted_loss
from optuna.trial import TrialState
import joblib
from optuna import TrialPruned
from optuna.storages import RDBStorage

seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train(model, loader, optimizer, criterion, device):
    model.train()
    #loss_batch = []
    for batch_x_acc, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, _ in loader:
        # 将数据移动到设备
        batch_x_acc = batch_x_acc.to(device)
        batch_x_att_continuous = batch_x_att_continuous.to(device)
        batch_x_att_discrete = batch_x_att_discrete.to(device)
        batch_y_HIC = batch_y_HIC.to(device)

        # 前向传播
        batch_pred_HIC, encoder_output, decoder_output = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)

        # 计算损失
        loss = criterion(batch_pred_HIC, batch_y_HIC)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        #loss_batch.append(loss.item())

    #return np.mean(loss_batch)


def valid(model, loader, criterion, device, y_transform=None):
    """
    参数:
        model: 教师模型实例。
        loader: 数据加载器。
        criterion: 损失函数。
        device: GPU 或 CPU。
        y_transform: 数据集中的HIC标签变换对象。若数据集中的HIC标签没有进行变换则为None。
    """
    model.eval()
    #loss_batch = []
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

            # # 计算损失
            # loss = criterion(batch_pred_HIC, batch_y_HIC)
            # loss_batch.append(loss.item())

            # 如果使用了 y_transform，需要将HIC 值反变换回原始范围以计算指标
            if y_transform is not None:
                batch_pred_HIC = y_transform.inverse(batch_pred_HIC)
                batch_y_HIC = y_transform.inverse(batch_y_HIC)

            # 记录预测值和真实值
            all_HIC_preds.append(batch_pred_HIC.cpu().numpy())
            all_HIC_trues.append(batch_y_HIC.cpu().numpy())
            all_AIS_trues.append(batch_y_AIS.cpu().numpy())

    # 计算平均损失
    #avg_loss = np.mean(loss_batch)

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

    #return avg_loss, accuracy, mae, rmse
    return accuracy, mae, rmse


# 定义目标函数
def objective(trial):
    # 超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 0.008, 0.015, step=0.001)  # 学习率
    #batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])       # 批量大小
    batch_size = 512
    weight_decay = trial.suggest_float("weight_decay", 2e-4, 8e-4, step=1e-4)   # 权重衰减

    Ksize_init = trial.suggest_int("Ksize_init", 4, 12, step=2)  # TCN 初始卷积核大小，必须是偶数
    Ksize_mid = trial.suggest_int("Ksize_mid", 3, 5, step=2)    # TCN 中间卷积核大小，必须是奇数
    num_blocks_of_tcn = trial.suggest_int("num_blocks_of_tcn", 2, 6, step=1)           # TCN 块数
    num_layers_of_mlpE = trial.suggest_int("num_layers_of_mlpE", 4, 5, step=1)         # MLP 编码器层数
    num_layers_of_mlpD = trial.suggest_int("num_layers_of_mlpD", 4, 5, step=1)         # MLP 解码器层数
    mlpE_hidden = trial.suggest_int("mlpE_hidden", 96, 256, step=32) # MLP 编码器隐藏层维度
    mlpD_hidden = trial.suggest_int("mlpD_hidden", 128, 256, step=128) # MLP 解码器隐藏层维度
    encoder_output_dim = trial.suggest_int("encoder_output_dim", 64, 96, step=32)  # 编码器输出维度
    decoder_output_dim = trial.suggest_int("decoder_output_dim", 16, 64, step=16)    # 解码器输出维度
    dropout = trial.suggest_float("dropout", 0.05, 0.25, step=0.05)                         # Dropout 概率
    #base_loss = trial.suggest_categorical("base_loss", ["mse", "mae"])         # 基础损失函数
    base_loss = "mae"
    weight_factor_classify = trial.suggest_float("weight_factor_classify", 1.4, 1.85, step=0.05)
    weight_factor_sample =trial.suggest_float("weight_factor_sample", 0.3, 0.8, step=0.1)
    Epochs = 500
    eta_min = 5e-6
    # 是否使用 HIC 标签变换对象（暂时不使用）
    lower_bound = 0
    upper_bound = 2500
    HIC_transform = None  # HIC 标签变换对象 或 SigmoidTransform(lower_bound, upper_bound)

    # 加载数据集对象
    dataset = CrashDataset(y_transform=HIC_transform)
    # 从data文件夹直接加载数据集
    if dataset.y_transform is None:
        train_dataset = torch.load("./data/train_dataset.pt")
        val_dataset = torch.load("./data/val_dataset.pt")
    else:
        train_dataset = torch.load("./data/train_dataset_ytrans.pt")
        val_dataset = torch.load("./data/val_dataset_ytrans.pt")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = models.TeacherModel(
        Ksize_init=Ksize_init,
        Ksize_mid=Ksize_mid,
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

    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = weighted_loss(base_loss=base_loss, weight_factor_classify=weight_factor_classify, weight_factor_sample=weight_factor_sample).to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=eta_min)

    val_maes = []
    val_accuracies = []
    val_rmses = []
    # 训练模型
    for epoch in range(Epochs//10*3):

        train(model, train_loader, optimizer, criterion, device)
        
        val_accuracy, val_mae, val_rmse = valid(model, val_loader, criterion, device, y_transform=dataset.y_transform)

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
study_file = "./runs/optuna_study_teacher3.pkl"
study_name = "teacher_model_optimization3"
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

# 设定寻优优先的超参组合
# with open("./runs/TeacherModel_Train_01102155/TrainingRecord.json", "r") as f:
#     data = json.load(f)
# known_params = {
#     "learning_rate": data["hyperparameters related to training"]["Learning_rate"],
#     "batch_size": data["hyperparameters related to training"]["Batch_size"],
#     "weight_decay": data["hyperparameters related to training"]["weight_decay"],
#     "base_loss": data["hyperparameters related to training"]["base_loss"],
#     "weight_factor_classify": data["hyperparameters related to training"]["weight_factor_classify"],
#     "weight_factor_sample": data["hyperparameters related to training"]["weight_factor_sample"],

#     "Ksize_init": data["hyperparameters related to model"]["Ksize_init"],
#     "Ksize_mid": data["hyperparameters related to model"]["Ksize_mid"],
#     "num_blocks_of_tcn": data["hyperparameters related to model"]["num_blocks_of_tcn"],
#     "num_layers_of_mlpE": data["hyperparameters related to model"]["num_layers_of_mlpE"],
#     "num_layers_of_mlpD": data["hyperparameters related to model"]["num_layers_of_mlpD"],
#     "mlpE_hidden": data["hyperparameters related to model"]["mlpE_hidden"],
#     "mlpD_hidden": data["hyperparameters related to model"]["mlpD_hidden"],
#     "encoder_output_dim": data["hyperparameters related to model"]["encoder_output_dim"],
#     "decoder_output_dim": data["hyperparameters related to model"]["decoder_output_dim"],
#     "dropout": data["hyperparameters related to model"]["dropout"]
# }

# user_attrs = {
#     "experiment_name": "initial_tuning",
#     "note": "This is a manually tuned trial."
# }
# study.enqueue_trial(known_params, user_attrs=user_attrs)
# known_params["weight_factor_classify"] = 1.2
# known_params["weight_factor_sample"] = 1.0
# study.enqueue_trial(known_params)

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