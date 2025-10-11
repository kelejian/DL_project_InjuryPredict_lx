# -*- coding: utf-8 -*-
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import os, json
import time
from datetime import datetime
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score

from utils import models
from utils.weighted_loss import weighted_loss
from utils.dataset_prepare import CrashDataset
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck 
from utils.set_random_seed import set_random_seed

set_random_seed()

# --- 合并了蒸馏训练和验证的统一函数 ---
def run_one_epoch(student_model, teacher_model, loader, criterion, device, distill_weights, optimizer=None):
    """
    执行一个完整的学生模型训练（带蒸馏）或验证周期。

    参数:
        student_model: 学生模型实例。
        teacher_model: 教师模型实例 (仅在训练时用于生成特征)。
        loader: 数据加载器。
        criterion: 损失函数。
        device: GPU 或 CPU。
        distill_weights (tuple): (encoder_weight, decoder_weight) 蒸馏损失的权重。
        optimizer (optional): 优化器。提供时为训练模式。

    返回:
        metrics (dict): 包含该周期所有指标的字典。
    """
    is_train = optimizer is not None
    if is_train:
        student_model.train()
        teacher_model.eval() # 教师模型始终处于评估模式
    else:
        student_model.eval()

    loss_batch = []
    regression_loss_batch = []  # 新增：记录回归损失
    distill_encoder_loss_batch, distill_decoder_loss_batch = [], []
    all_preds, all_trues = [], []
    all_true_ais_head, all_true_ais_chest, all_true_ais_neck = [], [], []
    all_true_mais = []
    
    # 根据模式选择是否启用梯度计算
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            (batch_x_acc, batch_x_att_continuous, batch_x_att_discrete,
             batch_y_HIC, batch_y_Dmax, batch_y_Nij,
             batch_ais_head, batch_ais_chest, batch_ais_neck, batch_y_MAIS) = [d.to(device) for d in batch]

            batch_y_true = torch.stack([batch_y_HIC, batch_y_Dmax, batch_y_Nij], dim=1)

            # 学生模型前向传播
            student_pred, student_encoder_output, student_decoder_output = student_model(batch_x_att_continuous, batch_x_att_discrete)

            # --- 损失计算 ---
            # 基础回归损失
            regression_loss = criterion(student_pred, batch_y_true)
            total_loss = regression_loss
            
            # 记录回归损失
            regression_loss_batch.append(regression_loss.item())

            # 计算蒸馏损失（训练和验证都计算，用于监控）
            with torch.no_grad():
                _, teacher_encoder_output, teacher_decoder_output = teacher_model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)
            
            distill_encoder_loss = nn.MSELoss()(student_encoder_output, teacher_encoder_output)
            distill_decoder_loss = nn.MSELoss()(student_decoder_output, teacher_decoder_output)
            
            # 记录蒸馏损失
            distill_encoder_loss_batch.append(distill_encoder_loss.item())
            distill_decoder_loss_batch.append(distill_decoder_loss.item())
            
            # 计算并添加蒸馏损失到总损失（仅在训练时）
            if is_train:
                encoder_w, decoder_w = distill_weights
                total_loss += encoder_w * distill_encoder_loss + decoder_w * distill_decoder_loss

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # 记录用于计算指标的值
            loss_batch.append(total_loss.item())
            all_preds.append(student_pred.detach().cpu().numpy())
            all_trues.append(batch_y_true.detach().cpu().numpy())
            all_true_ais_head.append(batch_ais_head.cpu().numpy())
            all_true_ais_chest.append(batch_ais_chest.cpu().numpy())
            all_true_ais_neck.append(batch_ais_neck.cpu().numpy())
            all_true_mais.append(batch_y_MAIS.cpu().numpy())

    # --- 指标计算部分 ---
    avg_loss = np.mean(loss_batch)
    avg_regression_loss = np.mean(regression_loss_batch)  # 新增
    avg_distill_encoder_loss = np.mean(distill_encoder_loss_batch)  # 新增
    avg_distill_decoder_loss = np.mean(distill_decoder_loss_batch)  # 新增
    
    preds, trues = np.concatenate(all_preds), np.concatenate(all_trues)
    pred_hic, pred_dmax, pred_nij = preds[:, 0], preds[:, 1], preds[:, 2]
    true_hic, true_dmax, true_nij = trues[:, 0], trues[:, 1], trues[:, 2]
    
    ais_head_pred, ais_chest_pred, ais_neck_pred = AIS_cal_head(pred_hic), AIS_cal_chest(pred_dmax), AIS_cal_neck(pred_nij)
    true_ais_head, true_ais_chest, true_ais_neck = np.concatenate(all_true_ais_head), np.concatenate(all_true_ais_chest), np.concatenate(all_true_ais_neck)
    true_mais = np.concatenate(all_true_mais)
    mais_pred = np.maximum.reduce([ais_head_pred, ais_chest_pred, ais_neck_pred])
    
    metrics = {
        'loss': avg_loss,
        'regression_loss': avg_regression_loss,  # 新增
        'distill_encoder_loss': avg_distill_encoder_loss,  # 新增
        'distill_decoder_loss': avg_distill_decoder_loss,  # 新增
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
    ''' 训练带知识蒸馏的学生模型 '''
    from torch.utils.tensorboard import SummaryWriter

    # 创建独立文件夹保存本次运行结果
    current_time = datetime.now().strftime("%m%d%H%M")
    run_dir = os.path.join("./runs", f"StudentModel_Distill_{current_time}")
    os.makedirs(run_dir, exist_ok=True)

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=run_dir)
    
    ############################################################################################
    ############################################################################################
    # 定义所有可调超参数
    # 1. 教师模型路径
    teacher_run_dir = ".\\runs\\TeacherModel_10111315" # <-- 教师模型运行目录
    teacher_model_name = "best_mais_accu.pth"

    # 2. 优化与训练相关
    Epochs = 1000
    Batch_size = 512
    Learning_rate = 0.024
    Learning_rate_min = 5e-7
    weight_decay = 2e-4
    Patience = 1000

    # 3. 损失函数相关
    base_loss = "mae"
    weight_factor_classify = 1.1
    weight_factor_sample = 0.5
    loss_weights = (0.2, 1.0, 20.0) # HIC, Dmax, Nij 各自损失的权重
    distill_encoder_weight = 0.4 # 编码器蒸馏损失权重
    distill_decoder_weight = 0.02  # 解码器蒸馏损失权重

    # 4. 学生模型结构相关 (encoder/decoder_output_dim 会被教师模型覆盖)
    num_layers_of_mlpE = 4
    num_layers_of_mlpD = 4
    mlpE_hidden = 160
    mlpD_hidden = 128
    dropout = 0.25
    ############################################################################################
    ############################################################################################

    if Patience > Epochs: Patience = Epochs
    
    # --- 加载教师模型超参数 ---
    teacher_model_path = os.path.join(teacher_run_dir, teacher_model_name)
    with open(os.path.join(teacher_run_dir, "TrainingRecord.json"), "r") as f:
        teacher_hyperparams = json.load(f)["hyperparameters"]["model"]
    
    # 加载数据集对象
    # dataset = CrashDataset()
    train_dataset = torch.load("./data/train_dataset.pt")
    val_dataset = torch.load("./data/val_dataset.pt")
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 初始化学生模型 (部分参数与教师模型对齐) ---
    student_model = models.StudentModel(
        num_classes_of_discrete=train_dataset.dataset.num_classes_of_discrete,
        num_layers_of_mlpE=num_layers_of_mlpE, num_layers_of_mlpD=num_layers_of_mlpD,
        mlpE_hidden=mlpE_hidden, mlpD_hidden=mlpD_hidden,
        encoder_output_dim=teacher_hyperparams["encoder_output_dim"],
        decoder_output_dim=teacher_hyperparams["decoder_output_dim"],
        dropout=dropout
    ).to(device)

    # --- 初始化教师模型 ---
    teacher_model = models.TeacherModel(**teacher_hyperparams, num_classes_of_discrete=train_dataset.dataset.num_classes_of_discrete).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path))
    teacher_model.eval()

    criterion = weighted_loss(base_loss, weight_factor_classify, weight_factor_sample, loss_weights)
    optimizer = optim.AdamW(student_model.parameters(), lr=Learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=Learning_rate_min)

    # 初始化跟踪变量
    val_loss_history, val_mais_accu_history, val_chest_accu_history, val_head_accu_history, val_neck_accu_history = [], [], [], [], []
    Best_val_loss = float('inf')
    Best_mais_accu, Best_chest_accu, Best_head_accu, Best_neck_accu = 0, 0, 0, 0
    Best_dmax_mae, Best_hic_mae, Best_nij_mae = float('inf'), float('inf'), float('inf')
    best_loss_epoch, best_MAIS_accu_epoch, best_dmax_epoch, best_nij_epoch, best_chest_epoch, best_hic_epoch, best_head_epoch, best_neck_epoch = 0, 0, 0, 0, 0, 0, 0, 0

    # 主训练循环
    for epoch in range(Epochs):
        epoch_start_time = time.time()
        distill_weights = (distill_encoder_weight, distill_decoder_weight)

        train_metrics = run_one_epoch(student_model, teacher_model, train_loader, criterion, device, distill_weights, optimizer=optimizer)
        val_metrics = run_one_epoch(student_model, teacher_model, val_loader, criterion, device, distill_weights, optimizer=None)
        
        val_loss_history.append(val_metrics['loss'])
        val_mais_accu_history.append(val_metrics['accu_mais'])
        val_head_accu_history.append(val_metrics['accu_head'])
        val_chest_accu_history.append(val_metrics['accu_chest'])
        val_neck_accu_history.append(val_metrics['accu_neck'])

        print(f"Epoch {epoch+1}/{Epochs} | Train Loss: {train_metrics['loss']:.3f} (Reg: {train_metrics['regression_loss']:.3f}, Enc: {train_metrics['distill_encoder_loss']:.3f}, Dec: {train_metrics['distill_decoder_loss']:.3f})")
        print(f"            | Val Loss: {val_metrics['loss']:.3f} (Reg: {val_metrics['regression_loss']:.3f}, Enc: {val_metrics['distill_encoder_loss']:.3f}, Dec: {val_metrics['distill_decoder_loss']:.3f}) | MAIS Acc: {val_metrics['accu_mais']:.2f}%")
        print(f"            | Head Acc: {val_metrics['accu_head']:.2f}%, Chest Acc: {val_metrics['accu_chest']:.2f}%, Neck Acc: {val_metrics['accu_neck']:.2f}%")
        
        scheduler.step()

        # TensorBoard 记录 (训练)
        writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
        writer.add_scalar("Loss/Train_Regression", train_metrics['regression_loss'], epoch)  # 新增
        writer.add_scalar("Loss/Train_Distill_Encoder", train_metrics['distill_encoder_loss'], epoch)  # 新增
        writer.add_scalar("Loss/Train_Distill_Decoder", train_metrics['distill_decoder_loss'], epoch)  # 新增
        writer.add_scalar("Accuracy_Train/MAIS", train_metrics['accu_mais'], epoch)
        writer.add_scalar("Accuracy_Train/Head", train_metrics['accu_head'], epoch)
        writer.add_scalar("Accuracy_Train/Chest", train_metrics['accu_chest'], epoch)
        writer.add_scalar("Accuracy_Train/Neck", train_metrics['accu_neck'], epoch)
        writer.add_scalar("MAE_Train/Train_HIC", train_metrics['mae_hic'], epoch)
        writer.add_scalar("MAE_Train/Train_Dmax", train_metrics['mae_dmax'], epoch)
        writer.add_scalar("MAE_Train/Train_Nij", train_metrics['mae_nij'], epoch)

        # TensorBoard 记录 (验证)
        writer.add_scalar("Loss/Val", val_metrics['loss'], epoch)
        writer.add_scalar("Loss/Val_Regression", val_metrics['regression_loss'], epoch)  # 新增
        writer.add_scalar("Loss/Val_Distill_Encoder", val_metrics['distill_encoder_loss'], epoch)  # 新增
        writer.add_scalar("Loss/Val_Distill_Decoder", val_metrics['distill_decoder_loss'], epoch)  # 新增
        writer.add_scalar("Accuracy_Val/MAIS", val_metrics['accu_mais'], epoch)
        writer.add_scalar("Accuracy_Val/Head", val_metrics['accu_head'], epoch)
        writer.add_scalar("Accuracy_Val/Chest", val_metrics['accu_chest'], epoch)
        writer.add_scalar("Accuracy_Val/Neck", val_metrics['accu_neck'], epoch)
        writer.add_scalar("MAE_Val/HIC", val_metrics['mae_hic'], epoch)
        writer.add_scalar("MAE_Val/Dmax", val_metrics['mae_dmax'], epoch)
        writer.add_scalar("MAE_Val/Nij", val_metrics['mae_nij'], epoch)

        model_save_configs = [
            # (metric_key, best_var_name, epoch_var_name, filename, format_str, compare_func)
            ('accu_mais', 'Best_mais_accu', 'best_MAIS_accu_epoch', 'best_mais_accu.pth', 'MAIS accuracy: {:.2f}%', max),
            ('mae_dmax', 'Best_dmax_mae', 'best_dmax_epoch', 'best_dmax_mae.pth', 'Dmax MAE: {:.3f}', min),
            ('accu_chest', 'Best_chest_accu', 'best_chest_epoch', 'best_chest_accu.pth', 'Chest Acc: {:.2f}%', max),
            ('mae_hic', 'Best_hic_mae', 'best_hic_epoch', 'best_hic_mae.pth', 'HIC MAE: {:.3f}', min),
            ('accu_head', 'Best_head_accu', 'best_head_epoch', 'best_head_accu.pth', 'Head Acc: {:.2f}%', max),
            ('mae_nij', 'Best_nij_mae', 'best_nij_epoch', 'best_nij_mae.pth', 'Nij MAE: {:.3f}', min),
            ('accu_neck', 'Best_neck_accu', 'best_neck_epoch', 'best_neck_accu.pth', 'Neck Acc: {:.2f}%', max),
        ]
        
        for metric_key, best_var, epoch_var, filename, format_str, compare_func in model_save_configs:
            current_value = val_metrics[metric_key]
            best_value = globals()[best_var]
            
            is_better = (compare_func == max and current_value > best_value) or \
                       (compare_func == min and current_value < best_value)
            
            if is_better:
                globals()[best_var] = current_value
                globals()[epoch_var] = epoch + 1
                torch.save(student_model.state_dict(), os.path.join(run_dir, filename))
                print(f"Best model saved with val {format_str.format(current_value)} at epoch {epoch+1}")

        # 跟踪最佳验证损失
        if val_metrics['loss'] < Best_val_loss:
            Best_val_loss, best_loss_epoch = val_metrics['loss'], epoch + 1

        # 早停逻辑
        if epoch > Epochs * 0.4 and len(val_loss_history) >= Patience:
            recent_losses = val_loss_history[-Patience:]
            recent_accu = val_mais_accu_history[-Patience:]
            recent_accu_chest = val_chest_accu_history[-Patience:]
            
            loss_no_improve = all(loss >= Best_val_loss for loss in recent_losses)
            accu_no_improve = all(accu <= Best_mais_accu for accu in recent_accu)
            chest_accu_no_improve = all(accu <= Best_chest_accu for accu in recent_accu_chest)

            if loss_no_improve and accu_no_improve and chest_accu_no_improve:
                print(f"Early Stop at epoch: {epoch+1}!")
                print(f"Best MAIS accuracy: {Best_mais_accu:.2f}% (at epoch {best_MAIS_accu_epoch})")
                print(f"Lowest Val Loss: {Best_val_loss:.3f} (at epoch {best_loss_epoch})")
                print(f"Lowest Dmax MAE: {Best_dmax_mae:.3f} (at epoch {best_dmax_epoch})")
                break

        print(f"            | Time: {time.time()-epoch_start_time:.2f}s")

    writer.close()

    # --- 类型转换函数 ---
    def convert_numpy_types(obj):
        """递归转换NumPy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # --- 完整记录超参数和最终结果 ---
    results = {
        "Trainset_size": len(train_dataset),
        "Valset_size": len(val_dataset),
        "hyperparameters": {
            "training": 
            {
                "Epochs": Epochs, 
                "Batch_size": Batch_size, 
                "Learning_rate": Learning_rate, 
                "Learning_rate_min": Learning_rate_min, 
                "weight_decay": weight_decay,
                "Patience": Patience},
            "loss": 
            {
                "base_loss": base_loss, 
                "weight_factor_classify": weight_factor_classify,
                "weight_factor_sample": weight_factor_sample,
                "loss_weights": loss_weights,
                "distill_encoder_weight": distill_encoder_weight,
                "distill_decoder_weight": distill_decoder_weight
            },
            "model": {
                "num_layers_of_mlpE": num_layers_of_mlpE, "num_layers_of_mlpD": num_layers_of_mlpD, 
                "mlpE_hidden": mlpE_hidden, 
                "mlpD_hidden": mlpD_hidden, 
                "encoder_output_dim": teacher_hyperparams["encoder_output_dim"], "decoder_output_dim": teacher_hyperparams["decoder_output_dim"], 
                "dropout": dropout
            }
        },
        "results": {
            "final_epoch": epoch + 1,
            "best_mais_accuracy": np.round(float(Best_mais_accu), 2),
            "best_mais_accuracy_epoch": int(best_MAIS_accu_epoch),
            "best_chest_accuracy": np.round(float(Best_chest_accu), 2),
            "best_chest_accuracy_epoch": int(best_chest_epoch),
            "best_head_accuracy": np.round(float(Best_head_accu), 2),
            "best_head_accuracy_epoch": int(best_head_epoch),
            "best_neck_accuracy": np.round(float(Best_neck_accu), 2),
            "best_neck_accuracy_epoch": int(best_neck_epoch),
            "best_dmax_mae": np.round(float(Best_dmax_mae), 2),
            "best_dmax_mae_epoch": int(best_dmax_epoch),
            "best_hic_mae": np.round(float(Best_hic_mae), 2),
            "best_hic_mae_epoch": int(best_hic_epoch),
            "best_nij_mae": np.round(float(Best_nij_mae), 3),
            "best_nij_mae_epoch": int(best_nij_epoch),

            "lowest_val_loss": np.round(float(Best_val_loss), 3),
            "lowest_val_loss_epoch": int(best_loss_epoch),

            "last_epoch_metrics": {
                "val_loss": np.round(float(val_metrics['loss']), 3),
                "val_regression_loss": np.round(float(val_metrics['regression_loss']), 3),  # 新增
                "val_distill_encoder_loss": np.round(float(val_metrics['distill_encoder_loss']), 3),  # 新增
                "val_distill_decoder_loss": np.round(float(val_metrics['distill_decoder_loss']), 3),  # 新增
                "accu_mais": np.round(float(val_metrics['accu_mais']), 2),
                "accu_head": np.round(float(val_metrics['accu_head']), 2),
                "accu_chest": np.round(float(val_metrics['accu_chest']), 2),
                "accu_neck": np.round(float(val_metrics['accu_neck']), 2),
                "mae_hic": np.round(float(val_metrics['mae_hic']), 2),
                "mae_dmax": np.round(float(val_metrics['mae_dmax']), 2),
                "mae_nij": np.round(float(val_metrics['mae_nij']), 3),
            }
        }
    }

    # 转换所有NumPy类型
    results = convert_numpy_types(results)

    with open(os.path.join(run_dir, "TrainingRecord.json"), "w") as f:
        json.dump(results, f, indent=4)