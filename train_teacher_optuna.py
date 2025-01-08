import optuna
import torch, os
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import random
from utils import models
from utils.dataset_prepare import CrashDataset
from utils.weighted_loss import weighted_loss
from optuna.trial import TrialState
import joblib  # 用于保存和加载 Optuna 研究
from optuna import TrialPruned
from optuna.storages import RDBStorage

seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# 定义目标函数
def objective(trial):
    # 超参数搜索空间
    lr_options = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    #learning_rate = trial.suggest_categorical("learning_rate", lr_options)  # 学习率
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)  # 学习率
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])       # 批量大小
    wd_options = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    #weight_decay = trial.suggest_categorical("weight_decay", wd_options)   # 权重衰减
    weight_decay = trial.suggest_float("weight_decay", 5e-5, 5e-2, log=True)   # 权重衰减

    Ksize_init = trial.suggest_categorical("Ksize_init", [4, 6, 8, 10, 12])  # TCN 初始卷积核大小，必须是偶数
    Ksize_mid = trial.suggest_categorical("Ksize_mid", [3, 5, 7])    # TCN 中间卷积核大小，必须是奇数
    num_blocks_of_tcn = trial.suggest_int("num_blocks_of_tcn", 2, 7)           # TCN 块数
    num_layers_of_mlpE = trial.suggest_int("num_layers_of_mlpE", 2, 6)         # MLP 编码器层数
    num_layers_of_mlpD = trial.suggest_int("num_layers_of_mlpD", 2, 6)         # MLP 解码器层数
    mlpE_hidden = trial.suggest_categorical("mlpE_hidden", [64, 96, 128, 192, 256]) # MLP 编码器隐藏层维度
    mlpD_hidden = trial.suggest_categorical("mlpD_hidden", [64, 96, 128, 192]) # MLP 解码器隐藏层维度
    encoder_output_dim = trial.suggest_categorical("encoder_output_dim", [64, 96, 128, 160])  # 编码器输出维度
    decoder_output_dim = trial.suggest_categorical("decoder_output_dim", [16])    # 解码器输出维度
    dropout = trial.suggest_float("dropout", 0.05, 0.5, step=0.05)                         # Dropout 概率
    weighted_loss_factor = trial.suggest_float("weighted_loss_factor", 1.0, 1.25, step=0.01)  # 加权损失系数
    base_loss = trial.suggest_categorical("base_loss", ["mse", "mae"])         # 基础损失函数

    # 是否使用 HIC 标签变换对象
    lower_bound = 0
    upper_bound = 2500
    HIC_transform = None  # HIC 标签变换对象 或 SigmoidTransform(lower_bound, upper_bound)
    
    # 加载数据集
    dataset = CrashDataset(y_transform=HIC_transform)
    train_size = 5000
    val_size = 500
    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])
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
    criterion = weighted_loss(base_loss, weighted_loss_factor).to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

    val_maes = []
    # 训练模型
    for epoch in range(150):
        model.train()
        for batch_x_acc, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, batch_y_AIS in train_loader:
            batch_x_acc = batch_x_acc.to(device)
            batch_x_att_continuous = batch_x_att_continuous.to(device)
            batch_x_att_discrete = batch_x_att_discrete.to(device)
            batch_y_HIC = batch_y_HIC.to(device)

            # 前向传播
            batch_pred_HIC, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)

            # 计算损失
            loss = criterion(batch_pred_HIC, batch_y_HIC)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证模型
        model.eval()
        all_HIC_preds = []
        all_HIC_trues = []
        with torch.no_grad():
            for batch_x_acc, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, _ in val_loader:
                batch_x_acc = batch_x_acc.to(device)
                batch_x_att_continuous = batch_x_att_continuous.to(device)
                batch_x_att_discrete = batch_x_att_discrete.to(device)
                batch_y_HIC = batch_y_HIC.to(device)

                # 前向传播
                batch_pred_HIC, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)
                all_HIC_preds.append(batch_pred_HIC.cpu().numpy())
                all_HIC_trues.append(batch_y_HIC.cpu().numpy())

        # 计算验证集 MAE
        HIC_preds = np.concatenate(all_HIC_preds)
        HIC_trues = np.concatenate(all_HIC_trues)
        val_mae = np.mean(np.abs(HIC_preds - HIC_trues))

        # 添加剪枝逻辑
        trial.report(val_mae, epoch)  # 报告当前 epoch 的验证集 MAE
        if trial.should_prune():  # 检查是否应该剪枝
            raise optuna.TrialPruned()  # 剪枝当前 trial

        # 记录验证集 MAE
        if epoch > 140:
            val_maes.append(val_mae)

        # 更新学习率
        scheduler.step()

    return np.mean(val_maes)

# 定义保存和加载的文件路径
study_file = "./runs/optuna_study_teacher.pkl"
db_path = "sqlite:///./runs/optuna_study.db"

# 定义 SQLite 数据库路径
storage = RDBStorage(db_path)

# 创建中位数剪枝器
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=20,  # 前几次试验不剪枝
    n_warmup_steps=50,     # 每个试验的前几步不剪枝
    interval_steps=5,     # 每隔几步评估一次剪枝条件
    n_min_trials=20        # 计算中位数所需的最小历史试验数量
)
# 检查是否存在已有的研究
try:
    study = optuna.load_study(
        study_name="teacher_model_optimization",  # 研究名称
        storage=storage
    )
    print("Loading existing study...")
except KeyError:
    print("Creating new study...")
    study = optuna.create_study(
        study_name="teacher_model_optimization",  # 研究名称
        direction="minimize",  # 目标是最小化验证集 MAE
        storage=storage,  # 使用 SQLite 存储
        pruner=pruner
    )

# 定义回调函数，定期保存研究结果
def save_study_callback(study, trial, freq=5):
    if trial.number % freq == 0:  # 每完成 5 个 trial 保存一次
        joblib.dump(study, study_file)
        print(f"Study saved at trial {trial.number}")

# 运行优化，设置 timeout 和回调函数
try:
    study.optimize(
        objective,
        n_trials=500,  # 总 trial 数
        callbacks=[save_study_callback]  # 每次 trial 完成后调用回调函数
    )
except Exception as e:  # 捕获其他异常
    print(f"Training interrupted due to: {str(e)}")
    joblib.dump(study, study_file)
finally:
    print("Study saved to:", study_file)

# 输出最佳超参数
print("=" * 50)
print("Best hyperparameters: ", study.best_params)
print("Best MAE: ", study.best_value)
print("=" * 50)