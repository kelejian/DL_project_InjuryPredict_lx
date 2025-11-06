# -*- coding: utf-8 -*-
"""
train_teacher_KFold.py

使用 K-Fold 交叉验证训练教师模型 (TeacherModel)。
加载由 dataset_prepare.py 生成的 train_dataset.pt 和 val_dataset.pt，
将它们合并后进行 K-Fold 划分，并在每个 fold 上独立训练和验证模型。
最终报告 K-Fold 的平均性能。
"""

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T' # 似乎是特定环境的设置
import warnings
warnings.filterwarnings('ignore')
import json
import time
from datetime import datetime
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset # 引入 Subset 和 ConcatDataset
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score
from sklearn.model_selection import StratifiedKFold # 引入 StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

# --- 从 utils 导入必要的模块 ---
# 确保此脚本与 utils 文件夹在同一项目结构下
from utils import models
from utils.weighted_loss import weighted_loss
from utils.dataset_prepare import CrashDataset # 需要导入以加载 .pt 文件
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck 
from utils.set_random_seed import GLOBAL_SEED, set_random_seed # 导入 GLOBAL_SEED

set_random_seed() # 设置全局随机种子

# --- run_one_epoch 函数 (与 train_teacher.py 完全相同) ---
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
    all_true_ais_head, all_true_ais_chest, all_true_ais_neck = [], [], []
    all_true_mais = []
    
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
        'accu_head': accuracy_score(true_ais_head, ais_head_pred) * 100,
        'accu_chest': accuracy_score(true_ais_chest, ais_chest_pred) * 100,
        'accu_neck': accuracy_score(true_ais_neck, ais_neck_pred) * 100,
        'accu_mais': accuracy_score(true_mais, mais_pred) * 100,
        'mae_hic': mean_absolute_error(true_hic, pred_hic), 'rmse_hic': root_mean_squared_error(true_hic, pred_hic),
        'mae_dmax': mean_absolute_error(true_dmax, pred_dmax), 'rmse_dmax': root_mean_squared_error(true_dmax, pred_dmax),
        'mae_nij': mean_absolute_error(true_nij, pred_nij), 'rmse_nij': root_mean_squared_error(true_nij, pred_nij),
    }
    return metrics

if __name__ == "__main__":
    
    # --- 1. 定义超参数 ---
    ############################################################################################
    ############################################################################################
    # 定义所有可调超参数
    # 1. 优化与训练相关
    Epochs = 350
    Batch_size = 512
    Learning_rate = 0.025
    Learning_rate_min = 1e-6
    weight_decay = 5e-4
    Patience = 1000 # 早停轮数
    
    # 2. 损失函数相关
    base_loss = "mae"
    weight_factor_classify = 1.3
    weight_factor_sample = 0.5
    loss_weights = (0.3, 1.0, 25.0) # HIC, Dmax, Nij 各自损失的权重

    # 3. 模型结构相关
    Ksize_init = 8
    Ksize_mid = 3
    num_blocks_of_tcn = 4
    tcn_channels_list = [64, 96, 128, 128]  # 每个 TCN 块的输出通道数
    num_layers_of_mlpE = 3
    num_layers_of_mlpD = 3
    mlpE_hidden = 192
    mlpD_hidden = 160
    encoder_output_dim = 128
    decoder_output_dim = 96
    dropout_MLP = 0.2
    dropout_TCN = 0.1
    use_channel_attention = True  # 是否使用通道注意力机制
    fixed_channel_weight = [0.7, 0.3, 0]  # 固定的通道注意力权重(None表示自适应学习)
    ############################################################################################
    ############################################################################################
    
    # K-Fold 设置
    K = 5 # 设置 K 值 (例如 5 或 10)
    
    # --- 2. 创建本次 K-Fold 运行的主目录 ---
    current_time = datetime.now().strftime("%m%d%H%M")
    main_run_dir = os.path.join("./runs", f"TeacherModel_KFold_{current_time}")
    os.makedirs(main_run_dir, exist_ok=True)
    print(f"K-Fold 主运行目录: {main_run_dir}")

    # --- 3. 加载由 dataset_prepare.py 生成的数据 ---
    print("正在加载 pt dataset ...")
    try:
        train_subset_orig = torch.load("./data/train_dataset.pt")
        val_subset_orig = torch.load("./data/val_dataset.pt")
        test_subset_orig = torch.load("./data/test_dataset.pt")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保 './data/' 目录下存在 train_dataset.pt 和 val_dataset.pt 文件。")
        print("您需要先运行 utils/dataset_prepare.py 来生成这些文件。")
        exit()
        
    # 获取底层的 CrashDataset 实例 (假设两个 Subset 指向同一个实例)
    full_processed_dataset = train_subset_orig.dataset
    
    # 合并训练集和验证集的【索引】用于 K-Fold 划分
    # combined_indices = np.concatenate([train_subset_orig.indices, val_subset_orig.indices])
    combined_indices = np.concatenate([train_subset_orig.indices, val_subset_orig.indices, test_subset_orig.indices])
    
    # 获取用于【分层】的标签 (从底层数据集中按合并后的索引提取)
    combined_labels = full_processed_dataset.mais[combined_indices]
    
    print(f"已加载并合并数据用于 K-Fold。总样本数: {len(combined_indices)}")
    
    # 获取模型所需的 num_classes_of_discrete
    num_classes_of_discrete = full_processed_dataset.num_classes_of_discrete
    
    # --- 4. 初始化 KFold ---
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=GLOBAL_SEED)
    
    # --- 5. 存储每一折的最佳验证指标 ---
    all_folds_best_metrics = [] # 存储每折的最佳 val_metrics 字典
    all_folds_best_epochs = []  # 存储每折达到最佳指标的 epoch
    
    # --- 6. K-Fold 交叉验证主循环 ---
    for fold, (train_k_indices, val_k_indices) in enumerate(skf.split(combined_indices, combined_labels)):
        
        fold_start_time = time.time()
        print("\n" + "="*50)
        print(f"                 Fold {fold+1}/{K}")
        print("="*50)
        
        # --- 6.1 创建当前 Fold 的运行目录和 TensorBoard Writer ---
        fold_run_dir = os.path.join(main_run_dir, f"Fold_{fold+1}")
        os.makedirs(fold_run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=fold_run_dir)
        
        # --- 6.2 获取当前 Fold 对应的【原始数据集索引】 ---
        # kf.split 返回的是 combined_indices 数组内部的索引，需要映射回 full_processed_dataset 的索引
        train_orig_indices = combined_indices[train_k_indices]
        val_orig_indices = combined_indices[val_k_indices]
        
        # --- 6.3 创建当前 Fold 的 Subset 和 DataLoader ---
        train_subset_k = Subset(full_processed_dataset, train_orig_indices)
        val_subset_k = Subset(full_processed_dataset, val_orig_indices)
        
        train_loader_k = DataLoader(train_subset_k, batch_size=Batch_size, shuffle=True, num_workers=0)
        val_loader_k = DataLoader(val_subset_k, batch_size=Batch_size, shuffle=False, num_workers=0)
        
        print(f"Fold {fold+1} 数据划分 - Train: {len(train_subset_k)}, Valid: {len(val_subset_k)}")
        
        # --- 6.4 **重新初始化模型、优化器、调度器** ---
        # (确保每折训练的独立性)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.TeacherModel(
            Ksize_init=Ksize_init, Ksize_mid=Ksize_mid,
            num_classes_of_discrete=num_classes_of_discrete,
            num_blocks_of_tcn=num_blocks_of_tcn, tcn_channels_list=tcn_channels_list,
            num_layers_of_mlpE=num_layers_of_mlpE, num_layers_of_mlpD=num_layers_of_mlpD,
            mlpE_hidden=mlpE_hidden, mlpD_hidden=mlpD_hidden,
            encoder_output_dim=encoder_output_dim, decoder_output_dim=decoder_output_dim,
            dropout_MLP=dropout_MLP, dropout_TCN=dropout_TCN,
            use_channel_attention=use_channel_attention, fixed_channel_weight=fixed_channel_weight
        ).to(device)

        # 在第一折打印: 模型各层参数量和总量, 我想知道模型参数量集中在哪里
        if fold == 0:
            print("\n模型结构:")
            print(model)
            print("\n模型各层参数量:")
            for name, param in model.named_parameters():
                print(f"  {name}: {param.numel()} parameters")
            print(f"\n模型参数量统计:")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"模型总参数量: {total_params}, 可训练参数量: {trainable_params}")
            
        
        criterion = weighted_loss(base_loss, weight_factor_classify, weight_factor_sample, loss_weights)
        optimizer = optim.AdamW(model.parameters(), lr=Learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)

        # --- 6.5 初始化当前 Fold 的跟踪变量 ---
        # (与 train_teacher.py 类似，但用于当前 Fold)
        val_loss_history, val_mais_accu_history = [], []
        best_fold_val_loss = float('inf')
        best_fold_mais_accu = 0
        best_fold_epoch = 0
        
        # --- 6.6 Epoch 训练循环 (内层循环，与 train_teacher.py 基本一致) ---
        if Patience > Epochs: current_patience = Epochs
        else: current_patience = Patience
            
        for epoch in range(Epochs):
            epoch_start_time = time.time()

            # --- 调用统一函数进行训练 ---
            train_metrics = run_one_epoch(model, train_loader_k, criterion, device, optimizer=optimizer)

            # --- 调用统一函数进行验证 ---
            val_metrics = run_one_epoch(model, val_loader_k, criterion, device, optimizer=None)
            
            val_loss_history.append(val_metrics['loss'])
            val_mais_accu_history.append(val_metrics['accu_mais'])

            # 打印当前 Fold 的 Epoch 信息
            print(f"  Epoch {epoch+1}/{Epochs} | Train Loss: {train_metrics['loss']:.3f} | Val Loss: {val_metrics['loss']:.3f} | Val MAIS Acc: {val_metrics['accu_mais']:.2f}% | Time: {time.time()-epoch_start_time:.2f}s")
            
            scheduler.step()

            # --- TensorBoard 记录 (与 train_teacher.py 类似) ---
            # 训练指标
            writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
            writer.add_scalar("Accuracy_Train/MAIS", train_metrics['accu_mais'], epoch)
            # ... 可添加其他训练指标 ...
            writer.add_scalar("MAE_Train/Train_HIC", train_metrics['mae_hic'], epoch)
            writer.add_scalar("MAE_Train/Train_Dmax", train_metrics['mae_dmax'], epoch)
            writer.add_scalar("MAE_Train/Train_Nij", train_metrics['mae_nij'], epoch)

            # 验证指标
            writer.add_scalar("Loss/Val", val_metrics['loss'], epoch)
            writer.add_scalar("Accuracy_Val/MAIS", val_metrics['accu_mais'], epoch)
            writer.add_scalar("Accuracy_Val/Head", val_metrics['accu_head'], epoch)
            writer.add_scalar("Accuracy_Val/Chest", val_metrics['accu_chest'], epoch)
            writer.add_scalar("Accuracy_Val/Neck", val_metrics['accu_neck'], epoch)
            writer.add_scalar("MAE_Val/HIC", val_metrics['mae_hic'], epoch)
            writer.add_scalar("MAE_Val/Dmax", val_metrics['mae_dmax'], epoch)
            writer.add_scalar("MAE_Val/Nij", val_metrics['mae_nij'], epoch)

            # --- 跟踪当前 Fold 的最佳模型 (以 MAIS 准确率为例) ---
            if val_metrics['accu_mais'] > best_fold_mais_accu:
                best_fold_mais_accu = val_metrics['accu_mais']
                best_fold_val_loss = val_metrics['loss'] # 记录此时的损失
                best_fold_epoch = epoch + 1
                best_fold_metrics_dict = val_metrics # 存储整个指标字典
                
                # 保存当前 Fold 的最佳模型权重
                torch.save(model.state_dict(), os.path.join(fold_run_dir, "best_mais_accu_model.pth"))
                print(f"    Best model for Fold {fold+1} saved with Val MAIS Acc: {best_fold_mais_accu:.2f}% at epoch {best_fold_epoch}")

            # --- 早停逻辑 (与 train_teacher.py 类似) ---
            if epoch > Epochs * 0.4 and len(val_loss_history) >= current_patience:
                # 简化：仅基于 MAIS 准确率是否连续 Patience 轮未超过最佳值
                recent_accu = val_mais_accu_history[-current_patience:]
                accu_no_improve = all(accu <= best_fold_mais_accu for accu in recent_accu)

                if accu_no_improve:
                    print(f"    Early Stop at epoch {epoch+1} for Fold {fold+1}!")
                    break # 跳出当前 Fold 的 Epoch 循环

        # --- 6.7 当前 Fold 训练结束 ---
        writer.close() # 关闭当前 Fold 的 writer
        print(f"Fold {fold+1} finished in {time.time() - fold_start_time:.2f}s.")
        print(f"  Best Val MAIS Accuracy for Fold {fold+1}: {best_fold_mais_accu:.2f}% (at epoch {best_fold_epoch})")
        
        # 记录当前 fold 的最佳结果
        all_folds_best_metrics.append(best_fold_metrics_dict)
        all_folds_best_epochs.append(best_fold_epoch)

    # --- 7. K-Fold 循环结束，计算并打印总体结果 ---
    print("\n" + "="*60)
    print("         K-Fold Cross-Validation Summary")
    print("="*60)
    
    # 将列表转换为 DataFrame 便于计算
    metrics_df = pd.DataFrame(all_folds_best_metrics)
    
    # 计算主要指标的均值和标准差
    mean_mais_acc = metrics_df['accu_mais'].mean()
    std_mais_acc = metrics_df['accu_mais'].std()
    mean_head_acc = metrics_df['accu_head'].mean()
    std_head_acc = metrics_df['accu_head'].std()
    mean_chest_acc = metrics_df['accu_chest'].mean()
    std_chest_acc = metrics_df['accu_chest'].std()
    mean_neck_acc = metrics_df['accu_neck'].mean()
    std_neck_acc = metrics_df['accu_neck'].std()
    
    mean_hic_mae = metrics_df['mae_hic'].mean()
    std_hic_mae = metrics_df['mae_hic'].std()
    mean_dmax_mae = metrics_df['mae_dmax'].mean()
    std_dmax_mae = metrics_df['mae_dmax'].std()
    mean_nij_mae = metrics_df['mae_nij'].mean()
    std_nij_mae = metrics_df['mae_nij'].std()
    
    mean_loss = metrics_df['loss'].mean()
    std_loss = metrics_df['loss'].std()
    
    print(f"Ran {K}-Fold Cross-Validation.")
    print(f"Average Best Epoch across folds: {np.mean(all_folds_best_epochs):.1f}")
    
    print("\n--- Average Validation Metrics (Mean +/- Std) ---")
    print(f"  Loss      : {mean_loss:.4f} +/- {std_loss:.4f}")
    print(f"  MAIS Acc  : {mean_mais_acc:.2f}% +/- {std_mais_acc:.2f}%")
    print(f"  Head Acc  : {mean_head_acc:.2f}% +/- {std_head_acc:.2f}%")
    print(f"  Chest Acc : {mean_chest_acc:.2f}% +/- {std_chest_acc:.2f}%")
    print(f"  Neck Acc  : {mean_neck_acc:.2f}% +/- {std_neck_acc:.2f}%")
    print(f"  HIC MAE   : {mean_hic_mae:.4f} +/- {std_hic_mae:.4f}")
    print(f"  Dmax MAE  : {mean_dmax_mae:.4f} +/- {std_dmax_mae:.4f}")
    print(f"  Nij MAE   : {mean_nij_mae:.4f} +/- {std_nij_mae:.4f}")
    
    print("="*60)
    
    # --- 8. 保存 K-Fold 总体结果 ---
    
    # --- 类型转换函数 (来自 train_teacher.py) ---
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
            
    kfold_results = {
        "model_type": "TeacherModel",
        "model_params": {
            "total_params": total_params,
            "trainable_params": trainable_params
        },
        "dataset_info": {
            "total_samples_for_kfold": len(combined_indices),
            "k_value": K
        },
        "hyperparameters": { # 记录使用的超参数
             "training": {
                "Epochs": Epochs, "Batch_size": Batch_size, "Learning_rate": Learning_rate,
                "Learning_rate_min": Learning_rate_min, "weight_decay": weight_decay,
                "Patience": Patience,
            },
            "loss": {
                "base_loss": base_loss, "weight_factor_classify": weight_factor_classify,
                "weight_factor_sample": weight_factor_sample, "loss_weights": loss_weights,
            },
            "model": {
                "Ksize_init": Ksize_init, "Ksize_mid": Ksize_mid, "num_blocks_of_tcn": num_blocks_of_tcn,
                "tcn_channels_list": tcn_channels_list,
                "num_layers_of_mlpE": num_layers_of_mlpE, "num_layers_of_mlpD": num_layers_of_mlpD,
                "mlpE_hidden": mlpE_hidden, "mlpD_hidden": mlpD_hidden,
                "encoder_output_dim": encoder_output_dim, "decoder_output_dim": decoder_output_dim,
                "dropout_MLP": dropout_MLP, "dropout_TCN": dropout_TCN,
                "use_channel_attention": use_channel_attention,
                "fixed_channel_weight": fixed_channel_weight
            }
        },
        "kfold_summary": {
            "mean_val_loss": mean_loss, "std_val_loss": std_loss,
            "mean_val_mais_acc": mean_mais_acc, "std_val_mais_acc": std_mais_acc,
            "mean_val_head_acc": mean_head_acc, "std_val_head_acc": std_head_acc,
            "mean_val_chest_acc": mean_chest_acc, "std_val_chest_acc": std_chest_acc,
            "mean_val_neck_acc": mean_neck_acc, "std_val_neck_acc": std_neck_acc,
            "mean_val_hic_mae": mean_hic_mae, "std_val_hic_mae": std_hic_mae,
            "mean_val_dmax_mae": mean_dmax_mae, "std_val_dmax_mae": std_dmax_mae,
            "mean_val_nij_mae": mean_nij_mae, "std_val_nij_mae": std_nij_mae,
            "mean_best_epoch": np.mean(all_folds_best_epochs)
        },
        "per_fold_best_metrics": convert_numpy_types(all_folds_best_metrics), # 记录每折的具体最佳指标
        "best_epochs_per_fold": all_folds_best_epochs
    }

    # 转换所有 NumPy 类型以确保 JSON 兼容性
    kfold_results = convert_numpy_types(kfold_results)

    # 保存到主运行目录
    results_path = os.path.join(main_run_dir, "KFold_TrainingRecord.json")
    with open(results_path, "w") as f:
        json.dump(kfold_results, f, indent=4)
        
    print(f"\nK-Fold 总体结果已保存至: {results_path}")