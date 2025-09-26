""" Test the TCN-based post-crash injury prediction model """
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os, json
import pandas as pd
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, root_mean_squared_error

from utils import models
from utils.dataset_prepare import CrashDataset, AIS_cal, AIS_3_cal
import warnings

warnings.filterwarnings('ignore')

from utils.set_random_seed import set_random_seed
set_random_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, loader):
    """
    参数:
        model: 模型实例。
        loader: 数据加载器。
    返回:
        HIC_preds: 预测的 HIC 值。
        HIC_trues: 真实的 HIC 值。
    """
    model.eval()
    all_HIC_preds = []
    all_HIC_trues = []

    with torch.no_grad():
        for batch_x_acc, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, batch_y_AIS in loader:
            # 将数据移动到设备
            batch_x_acc = batch_x_acc.to(device)
            batch_x_att_continuous = batch_x_att_continuous.to(device)
            batch_x_att_discrete = batch_x_att_discrete.to(device)
            batch_y_HIC = batch_y_HIC.to(device)

            # 前向传播
            if isinstance(model, models.TeacherModel):
                batch_pred_HIC, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)
            elif isinstance(model, models.StudentModel):
                batch_pred_HIC, _, _ = model(batch_x_att_continuous, batch_x_att_discrete)

            # 记录预测值和真实值
            all_HIC_preds.append(batch_pred_HIC.cpu().numpy())
            all_HIC_trues.append(batch_y_HIC.cpu().numpy())
    
    # 拼接所有 batch 的结果
    HIC_preds = np.concatenate(all_HIC_preds)
    HIC_trues = np.concatenate(all_HIC_trues)

    # 检查 HIC_preds 和 HIC_trues 是否包含 inf 或异常值
    if np.isinf(HIC_preds).any() or np.isinf(HIC_trues).any():
        print("**Warning: HIC_preds or HIC_trues contains inf values. Replacing inf with large finite values.**")
        HIC_preds = np.nan_to_num(HIC_preds, nan=0.0, posinf=1e4, neginf=-1e4)
        HIC_trues = np.nan_to_num(HIC_trues, nan=0.0, posinf=1e4, neginf=-1e4)

    return HIC_preds, HIC_trues

if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="Test Teacher or Student Model")
    parser.add_argument("--run_dir", '-r', type=str, default=".\\runs\\StudentModel_Baseline_01122201")
    parser.add_argument("--weight_file", '-w', type=str, default="student_best_mae.pth")
    args = parser.parse_args()

    # 加载超参数和训练记录
    with open(os.path.join(args.run_dir, "TrainingRecord.json"), "r") as f:
        training_record = json.load(f)

    # 提取模型相关的超参数
    model_params = training_record["hyperparameters related to model"]
    num_layers_of_mlpE = model_params["num_layers_of_mlpE"]
    num_layers_of_mlpD = model_params["num_layers_of_mlpD"]
    mlpE_hidden = model_params["mlpE_hidden"]
    mlpD_hidden = model_params["mlpD_hidden"]
    encoder_output_dim = model_params["encoder_output_dim"]
    decoder_output_dim = model_params["decoder_output_dim"]
    dropout = model_params["dropout"]

    # 提取训练相关的超参数
    train_params = training_record["hyperparameters related to training"]

    # 加载数据集
    dataset = CrashDataset()

    test_dataset1 = torch.load("./data/val_dataset.pt")
    test_dataset2 = torch.load("./data/test_dataset.pt")
        
    test_dataset = ConcatDataset([test_dataset1, test_dataset2])
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 判断是教师模型还是学生模型
    if "teacher" in args.weight_file.lower():
        Ksize_init = model_params["Ksize_init"]
        Ksize_mid = model_params["Ksize_mid"]
        num_blocks_of_tcn = model_params.get("num_blocks_of_tcn", None)  # 仅教师模型需要
        # 加载教师模型
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
        
    elif "student" in args.weight_file.lower():
        # 加载学生模型
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
    else:
        raise ValueError("Weight file name must contain 'teacher' or 'student' to identify the model type.")
    
    model.load_state_dict(torch.load(os.path.join(args.run_dir, args.weight_file)))

    HIC_preds, HIC_trues = test(model, test_loader)

    # 找出预测差别最大的5条样本，打印其预测值和真实值
    diff = np.abs(HIC_preds - HIC_trues)
    max_diff_indices = np.argsort(diff)[-5:][::-1]
    print("Top 5 samples with largest prediction errors:")
    for i in max_diff_indices:
        print(f"Sample {i}: Prediction: {HIC_preds[i]:.2f}, Ground Truth: {HIC_trues[i]:.2f}")


    HIC_preds[HIC_preds > 2500] = 2500 # 规定上界，过高的HIC值(>2000基本就危重伤)不具有实际意义
    HIC_trues[HIC_trues > 2500] = 2500  

    # HIC评估指标
    rmse = root_mean_squared_error(HIC_trues, HIC_preds)
    mae = mean_absolute_error(HIC_trues, HIC_preds)
    r2 = r2_score(HIC_trues, HIC_preds)
    print("均方根误差 (RMSE):", rmse)
    print("平均绝对误差 (MAE):", mae)
    print("决定系数 (R2):", r2)

    # AIS-6C评估指标
    pred, true = AIS_cal(HIC_preds), AIS_cal(HIC_trues)
    accu_6c = 100. * (1 - np.count_nonzero(true - pred) / float(len(true)))
    conf_mat_6c = confusion_matrix(true, pred)
    G_mean_6c = geometric_mean_score(true, pred)
    report_6c = classification_report_imbalanced(true, pred, digits=3)
    print('AIS-6C Accuracy: ' + str(np.around(accu_6c, 1)) + '%')
    print(conf_mat_6c)
    print(G_mean_6c)
    print(report_6c)

    # AIS-3C评估指标
    pred, true = AIS_3_cal(HIC_preds), AIS_3_cal(HIC_trues)
    accu_3c = 100. * (1 - np.count_nonzero(true - pred) / float(len(true)))
    conf_mat_3c = confusion_matrix(true, pred)
    G_mean_3c = geometric_mean_score(true, pred)
    report_3c = classification_report_imbalanced(true, pred, digits=3)
    print('AIS-3C Accuracy: ' + str(np.around(accu_3c, 1)) + '%')
    print(conf_mat_3c)
    print(G_mean_3c)
    print(report_3c)

    # 写入 Markdown 文件
    markdown_content = f"""
## Model Identification
- **Weight file**: {args.weight_file}
- **Model Directory**: {args.run_dir}

## HIC Metrics
- **RMSE**: {rmse:.4f}
- **MAE**: {mae:.4f}
- **R2**: {r2:.4f}

## AIS-6C Metrics
- **Accuracy**: {accu_6c:.2f}%
- **G-Mean**: {G_mean_6c:.4f}
- **Confusion Matrix**:
```
{conf_mat_6c}
```
- **Classification Report**:
```
{report_6c}
```

## AIS-3C Metrics
- **Accuracy**: {accu_3c:.2f}%
- **G-Mean**: {G_mean_3c:.4f}
- **Confusion Matrix**:
```
{conf_mat_3c}
```
- **Classification Report**:
```
{report_3c}
```
"""

    # 写入 Markdown 文件
    markdown_file = os.path.join(args.run_dir, f"TestResults_{args.weight_file.replace('.pth', '')}.md")
    with open(markdown_file, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)

    print(f"Results written to {markdown_file}")


    # 混淆矩阵画成图
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat_6c, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for AIS-6C', fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(tick_marks, [0, 1, 2, 3, 4, 5], fontsize=12)
    plt.yticks(tick_marks, [0, 1, 2, 3, 4, 5], fontsize=12)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    thresh = conf_mat_6c.max() / 2.
    for i, j in np.ndindex(conf_mat_6c.shape):
        plt.text(j, i, format(conf_mat_6c[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_mat_6c[i, j] > thresh else "black",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "AIS6C_confusion_matrix.png"))
    plt.close()

    # 绘制散点图
    plt.figure(figsize=(8, 7))
    
    # 定义颜色映射
    colors = ['blue', 'green', 'yellow', 'orange', 'red', 'darkred']
    AIS_trues = AIS_cal(HIC_trues)
    ais_colors = [colors[min(ais, 5)] for ais in AIS_trues]

    plt.scatter(HIC_trues, HIC_preds, c=ais_colors, alpha=0.5)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=f'AIS {i}') for i in range(6) if i in AIS_trues]
    plt.legend(handles=legend_elements, title='AIS LEVEL', loc='upper left')

    # plt.legend(fontsize=14)
    plt.plot([0, 2500], [0, 2500], 'r--', label="Ideal Line")  # 理想预测线即 y = x
    plt.xlabel("Ground Truth (HIC)", fontsize=16)
    plt.ylabel("Predictions (HIC)", fontsize=16)
    plt.title("Scatter Plot of Predictions vs Ground Truth On Test Set") 
    first_legend = plt.legend(handles=legend_elements, title='AIS LEVEL', loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.legend(loc='lower right')

    plt.grid(True)
    plt.savefig(os.path.join(args.run_dir, "HIC_scatter_plot.png"))
    plt.show()
    
