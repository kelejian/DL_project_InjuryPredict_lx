import optuna
import torch, os, json
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import random
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score
from optuna.storages import RDBStorage
import joblib

from utils import models
from utils.dataset_prepare import CrashDataset
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck 
from utils.weighted_loss import weighted_loss
from utils.set_random_seed import set_random_seed

set_random_seed()

# --- 新增: 采用与 train_teacher.py 一致的统一评估函数 ---
def run_one_epoch(model, loader, criterion, device, optimizer=None):
    """
    执行一个完整的训练或验证周期。
    """
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    loss_batch = []
    all_preds, all_trues = [], []
    all_true_ais_head, all_true_ais_chest, all_true_ais_neck, all_true_mais = [], [], [], []
    
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            (batch_x_acc, batch_x_att_continuous, batch_x_att_discrete,
             batch_y_HIC, batch_y_Dmax, batch_y_Nij,
             batch_ais_head, batch_ais_chest, batch_ais_neck, batch_y_MAIS) = [d.to(device) for d in batch]

            batch_y_true = torch.stack([batch_y_HIC, batch_y_Dmax, batch_y_Nij], dim=1)
            batch_pred, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)
            loss = criterion(batch_pred, batch_y_true)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_batch.append(loss.item())
            all_preds.append(batch_pred.detach().cpu().numpy())
            all_trues.append(batch_y_true.detach().cpu().numpy())
            all_true_ais_head.append(batch_ais_head.cpu().numpy())
            all_true_ais_chest.append(batch_ais_chest.cpu().numpy())
            all_true_ais_neck.append(batch_ais_neck.cpu().numpy())
            all_true_mais.append(batch_y_MAIS.cpu().numpy())

    avg_loss = np.mean(loss_batch)
    preds, trues = np.concatenate(all_preds), np.concatenate(all_trues)
    pred_hic, pred_dmax, pred_nij = preds[:, 0], preds[:, 1], preds[:, 2]
    true_hic, true_dmax, true_nij = trues[:, 0], trues[:, 1], trues[:, 2]
    
    ais_head_pred, ais_chest_pred, ais_neck_pred = AIS_cal_head(pred_hic), AIS_cal_chest(pred_dmax), AIS_cal_neck(pred_nij)
    true_ais_head, true_ais_chest, true_ais_neck = np.concatenate(all_true_ais_head), np.concatenate(all_true_ais_chest), np.concatenate(all_true_ais_neck)
    true_mais = np.concatenate(all_true_mais)
    mais_pred = np.maximum.reduce([ais_head_pred, ais_chest_pred, ais_neck_pred])
    
    metrics = {
        'loss': avg_loss,
        'accu_mais': accuracy_score(true_mais, mais_pred) * 100,
        'mae_hic': mean_absolute_error(true_hic, pred_hic),
        'mae_dmax': mean_absolute_error(true_dmax, pred_dmax),
        'mae_nij': mean_absolute_error(true_nij, pred_nij),
    }
    return metrics


def objective(trial):
    """定义Optuna的优化目标函数"""
    # --- 超参数搜索空间 ---
    # 优化与训练相关
    learning_rate = trial.suggest_float("learning_rate", 0.008, 0.03, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    # 损失函数相关
    weight_factor_classify = trial.suggest_float("weight_factor_classify", 1.2, 4.0)
    weight_factor_sample = trial.suggest_float("weight_factor_sample", 0.2, 1.0)
    
    # 模型结构相关
    Ksize_init = trial.suggest_int("Ksize_init", 4, 12, step=2)
    Ksize_mid = trial.suggest_categorical("Ksize_mid", [3, 5])
    num_blocks_of_tcn = trial.suggest_int("num_blocks_of_tcn", 2, 5)
    num_layers_of_mlpE = trial.suggest_int("num_layers_of_mlpE", 3, 5)
    num_layers_of_mlpD = trial.suggest_int("num_layers_of_mlpD", 3, 5)
    mlpE_hidden = trial.suggest_int("mlpE_hidden", 96, 256, step=32)
    mlpD_hidden = trial.suggest_int("mlpD_hidden", 96, 256, step=32)
    encoder_output_dim = trial.suggest_categorical("encoder_output_dim", [64, 96, 128])
    decoder_output_dim = trial.suggest_categorical("decoder_output_dim", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    
    # 固定参数
    Epochs = 150 # Optuna中通常使用较少的Epochs进行快速评估
    Batch_size = 512
    base_loss = "mae"
    loss_weights = (1.0, 1.0, 1.0)
    eta_min = 1e-6
    
    # 加载数据集
    dataset = CrashDataset()
    train_dataset = torch.load("./data/train_dataset.pt")
    val_dataset = torch.load("./data/val_dataset.pt")
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = models.TeacherModel(
        Ksize_init=Ksize_init, Ksize_mid=Ksize_mid, num_classes_of_discrete=dataset.num_classes_of_discrete,
        num_blocks_of_tcn=num_blocks_of_tcn, num_layers_of_mlpE=num_layers_of_mlpE,
        num_layers_of_mlpD=num_layers_of_mlpD, mlpE_hidden=mlpE_hidden, mlpD_hidden=mlpD_hidden,
        encoder_output_dim=encoder_output_dim, decoder_output_dim=decoder_output_dim, dropout=dropout
    ).to(device)

    # 定义优化器和损失函数
    criterion = weighted_loss(base_loss, weight_factor_classify, weight_factor_sample, loss_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=eta_min)

    val_metrics_list = []
    # 训练与验证循环
    for epoch in range(Epochs):
        run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_one_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        # Optuna剪枝逻辑 (可选)
        trial.report(val_metrics['loss'], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # 记录最后10轮的验证指标
        if epoch >= Epochs - 10:
            val_metrics_list.append(val_metrics)

    # --- 返回多目标优化的平均值 ---
    avg_mais_accu = np.mean([m['accu_mais'] for m in val_metrics_list])
    avg_mae_hic = np.mean([m['mae_hic'] for m in val_metrics_list])
    avg_mae_dmax = np.mean([m['mae_dmax'] for m in val_metrics_list])
    avg_mae_nij = np.mean([m['mae_nij'] for m in val_metrics_list])

    return avg_mais_accu, avg_mae_hic, avg_mae_dmax, avg_mae_nij

if __name__ == "__main__":
    study_file = "./runs/optuna_study_teacher_multiobj.pkl"
    study_name = "teacher_model_multiobj_optimization"
    db_path = "sqlite:///./runs/optuna_study.db"
    storage = RDBStorage(db_path)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Successfully loaded study '{study_name}' from the database.")
    except Exception as e:
        print(f"Failed to load study: {e}")
        print("Creating new multi-objective study...")
        # --- 修改: 配置多目标优化 ---
        study = optuna.create_study(
            sampler=optuna.samplers.NSGAIISampler(), # 推荐用于多目标优化的采样器
            study_name=study_name, 
            storage=storage, 
            directions=["maximize", "minimize", "minimize", "minimize"] # 对应MAIS Acc, HIC MAE, Dmax MAE, Nij MAE
        )

    # 定义回调函数，定期保存研究结果
    def save_study_callback(study, trial):
        if trial.number % 5 == 0:  
            joblib.dump(study, study_file)
            print(f"\nStudy saved to {study_file} at trial {trial.number}\n")

    # 运行优化
    try:
        study.optimize(objective, n_trials=100, callbacks=[save_study_callback])
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    finally:
        joblib.dump(study, study_file)
        print(f"Final study state saved to {study_file}")

    # --- 修改: 打印Pareto前沿的结果 ---
    print("\n" + "="*50)
    print("                 Pareto Front Results")
    print("="*50)
    for trial in study.best_trials:
        print(f"Trial Number: {trial.number}")
        print(f"  - MAIS Acc: {trial.values[0]:.2f}%")
        print(f"  - HIC MAE:  {trial.values[1]:.2f}")
        print(f"  - Dmax MAE: {trial.values[2]:.2f}")
        print(f"  - Nij MAE:  {trial.values[3]:.2f}")
        print(f"  - Hyperparameters: {json.dumps(trial.params, indent=4)}")
        print("-" * 50)