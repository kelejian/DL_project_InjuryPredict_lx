""" Test the TCN-based pre-crash injury prediction model, i.e., the student model with KD. """
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import  DataLoader
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, root_mean_squared_error

from utils import models
from utils.dataset_prepare import CrashDataset, AIS_cal, AIS_3_cal
import warnings

warnings.filterwarnings('ignore')

# Define the random seed.
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, loader):
    model.eval()
    all_HIC_preds = []
    all_HIC_trues = []
    
    with torch.no_grad():
        for batch_x_acc, batch_x_att, batch_y_HIC, batch_y_AIS in loader:
            batch_x_att = batch_x_att.to(device)
            batch_pred_HIC, _, _ = model(batch_x_att)
            all_HIC_preds.append(batch_pred_HIC.cpu().numpy())
            all_HIC_trues.append(batch_y_HIC.numpy())

    HIC_preds = np.concatenate(all_HIC_preds)
    HIC_trues = np.concatenate(all_HIC_trues)

    return HIC_preds, HIC_trues

if __name__ == "__main__":
    # 超参数设置
    Emb_size = 128
    Batch_size = 64

    # 数据加载
    dataset = CrashDataset()
    test_dataset1 = torch.load("./data/val_dataset.pt")
    test_dataset2 = torch.load("./data/test_dataset.pt")
    test_dataset = torch.utils.data.ConcatDataset([test_dataset1, test_dataset2])
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

    # 模型加载
    model = models.student_model(Emb_size).to(device)
    model.load_state_dict(torch.load("./ckpt/student_w_KD_best.pth"))

    # 测试模型
    HIC_preds, HIC_trues = test(model, test_loader)
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
# Student Model with KD Results

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

    file_path = "./results/student_model_w_KD_results.md"
    with open(file_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)

    print(f"Results written to {file_path}")


    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(HIC_trues, HIC_preds, c='blue', alpha=0.5, label="Predictions vs Ground Truth")
    plt.plot([0, 2500], [0, 2500], 'r--', label="Ideal Line")
    plt.xlabel("Ground Truth (HIC)")
    plt.ylabel("Predictions (HIC)")
    plt.title("Scatter Plot of Predictions vs Ground Truth On Test Set (Student Model with KD)") 
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/student_model_w_KD.png')
    plt.show()
    
