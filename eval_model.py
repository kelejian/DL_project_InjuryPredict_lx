"""
对教师或学生模型在测试集上的性能进行全面评估。
功能包括：
1. 计算三个损伤部位（头、胸、颈）的回归指标 (MAE, RMSE, R^2)。
2. 计算对应AIS等级以及MAIS的分类指标 (Accuracy, G-mean, Confusion Matrix, Report)。
3. 为HIC额外计算AIS-3C的分类指标。
4. 生成并保存在指定运行目录下的详细评估报告 (Markdown格式)。
5. 生成并保存所有损伤指标的散点图和所有AIS分类的混淆矩阵图。
"""
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os, json
import pandas as pd
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from utils import models
from utils.dataset_prepare import CrashDataset
from utils.AIS_cal import AIS_3_cal_head, AIS_cal_head, AIS_cal_chest, AIS_cal_neck 

from utils.set_random_seed import set_random_seed
set_random_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, loader):
    """
    在测试集上运行模型并收集所有预测和真实标签。

    返回:
        preds (np.ndarray): 模型对 [HIC, Dmax, Nij] 的预测值, 形状 (N, 3)。
        trues (dict): 包含所有真实标签的字典。
    """
    model.eval()
    all_preds = []
    all_trues_regression = []
    all_true_ais_head, all_true_ais_chest, all_true_ais_neck, all_true_mais = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            (batch_x_acc, batch_x_att_continuous, batch_x_att_discrete,
             batch_y_HIC, batch_y_Dmax, batch_y_Nij,
             batch_ais_head, batch_ais_chest, batch_ais_neck, batch_y_MAIS) = [d.to(device) for d in batch]
            
            # 前向传播
            if isinstance(model, models.TeacherModel):
                batch_pred, _, _ = model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)
            elif isinstance(model, models.StudentModel):
                batch_pred, _, _ = model(batch_x_att_continuous, batch_x_att_discrete)

            # 收集回归和分类的标签
            batch_y_true = torch.stack([batch_y_HIC, batch_y_Dmax, batch_y_Nij], dim=1)
            all_preds.append(batch_pred.cpu().numpy())
            all_trues_regression.append(batch_y_true.cpu().numpy())
            all_true_ais_head.append(batch_ais_head.cpu().numpy())
            all_true_ais_chest.append(batch_ais_chest.cpu().numpy())
            all_true_ais_neck.append(batch_ais_neck.cpu().numpy())
            all_true_mais.append(batch_y_MAIS.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    trues = {
        'regression': np.concatenate(all_trues_regression),
        'ais_head': np.concatenate(all_true_ais_head),
        'ais_chest': np.concatenate(all_true_ais_chest),
        'ais_neck': np.concatenate(all_true_ais_neck),
        'mais': np.concatenate(all_true_mais)
    }
    
    return preds, trues

def get_regression_metrics(y_true, y_pred):
    """计算并返回一组回归指标"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def get_classification_metrics(y_true, y_pred, labels):
    """计算并返回一组分类指标 - 改进版"""
    # 检查缺失的类别
    present_labels = set(np.unique(np.concatenate([y_true, y_pred])))
    missing_labels = set(labels) - present_labels
    
    if missing_labels:
        print(f"\n*Warning: Labels {missing_labels} are not present in the data\n")
    
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'g_mean': geometric_mean_score(y_true, y_pred, labels=labels),
        'conf_matrix': confusion_matrix(y_true, y_pred, labels=labels),
        'report': classification_report_imbalanced(
            y_true, y_pred, labels=labels, digits=3, 
            zero_division=0  # 处理除零情况
        )
    }

def plot_scatter(y_true, y_pred, ais_true, title, xlabel, save_path):
    """改进的散点图函数"""
    plt.figure(figsize=(8, 7))
    colors = ['blue', 'green', 'yellow', 'orange', 'red', 'darkred']
    ais_colors = [colors[min(ais, 5)] for ais in ais_true]
    plt.scatter(y_true, y_pred, c=ais_colors, alpha=0.5, s=40)

    # 显示所有可能的类别，即使数据中不存在
    # all_possible_ais = range(6)
    # legend_elements = [
    #     Patch(facecolor=colors[i], 
    #           label=f'AIS {i}' + (' (absent)' if i not in np.unique(ais_true) else ''))
    #     for i in all_possible_ais
    # ]

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=f'AIS {i}') for i in range(6) if i in np.unique(ais_true)]
    
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05
    plt.plot([0, max_val], [0, max_val], 'r--', label="Ideal Line")
    plt.xlabel(f"Ground Truth ({xlabel})", fontsize=16)
    plt.ylabel(f"Predictions ({xlabel})", fontsize=16)
    plt.title(f"Scatter Plot of Predictions vs Ground Truth\n({title})", fontsize=18)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    first_legend = plt.legend(handles=legend_elements, title='AIS Level', loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, labels, title, save_path):
    """绘制并保存混淆矩阵图"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_report_section(title, reg_metrics, cls_metrics_6c, cls_metrics_3c=None):
    """生成Markdown报告的一个区域"""
    section = f"## {title} Metrics\n\n"
    section += f"- **MAE**: {reg_metrics['mae']:.4f}\n"
    section += f"- **RMSE**: {reg_metrics['rmse']:.4f}\n"
    section += f"- **R² Score**: {reg_metrics['r2']:.4f}\n\n"
    
    section += f"### AIS-6C/5C Classification\n\n"
    section += f"- **Accuracy**: {cls_metrics_6c['accuracy']:.2f}%\n"
    section += f"- **G-Mean**: {cls_metrics_6c['g_mean']:.4f}\n"
    section += f"- **Confusion Matrix**:\n```\n{cls_metrics_6c['conf_matrix']}\n```\n"
    section += f"- **Classification Report**:\n```\n{cls_metrics_6c['report']}\n```\n"
    
    if cls_metrics_3c:
        section += f"### AIS-3C Classification (HIC only)\n\n"
        section += f"- **Accuracy**: {cls_metrics_3c['accuracy']:.2f}%\n"
        section += f"- **G-Mean**: {cls_metrics_3c['g_mean']:.4f}\n"
        section += f"- **Confusion Matrix**:\n```\n{cls_metrics_3c['conf_matrix']}\n```\n"
        section += f"- **Classification Report**:\n```\n{cls_metrics_3c['report']}\n```\n"
    return section

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained injury prediction model")
    parser.add_argument("--run_dir", '-r', type=str, required=True, help="Directory of the training run to evaluate.")
    parser.add_argument("--weight_file", '-w', type=str, default="best_mais_accu.pth", help="Name of the model weight file.")
    args = parser.parse_args()

    # --- 1. 加载模型和数据 ---
    with open(os.path.join(args.run_dir, "TrainingRecord.json"), "r") as f:
        training_record = json.load(f)
    
    model_params = training_record["hyperparameters"]["model"]
    
    dataset = CrashDataset()
    test_dataset1 = torch.load("./data/val_dataset.pt")
    test_dataset2 = torch.load("./data/test_dataset.pt")
    test_dataset = ConcatDataset([test_dataset1, test_dataset2])
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    if "teacher" in args.run_dir.lower():
        model = models.TeacherModel(**model_params, num_classes_of_discrete=dataset.num_classes_of_discrete).to(device)
    elif "student" in args.run_dir.lower():
        model = models.StudentModel(**model_params, num_classes_of_discrete=dataset.num_classes_of_discrete).to(device)
    else:
        raise ValueError("run_dir name must contain 'teacher' or 'student'.")
    
    model.load_state_dict(torch.load(os.path.join(args.run_dir, args.weight_file)))

    # --- 2. 执行预测 ---
    predictions, ground_truths = test(model, test_loader)
    
    pred_hic, pred_dmax, pred_nij = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    true_hic, true_dmax, true_nij = ground_truths['regression'][:, 0], ground_truths['regression'][:, 1], ground_truths['regression'][:, 2]

    # --- 3. 计算所有指标 ---
    # 回归指标
    reg_metrics_hic = get_regression_metrics(true_hic, pred_hic)
    reg_metrics_dmax = get_regression_metrics(true_dmax, pred_dmax)
    reg_metrics_nij = get_regression_metrics(true_nij, pred_nij)

    # 分类指标 
    cls_metrics_head = get_classification_metrics(ground_truths['ais_head'], AIS_cal_head(pred_hic),  list(range(6)))
    cls_metrics_chest = get_classification_metrics(ground_truths['ais_chest'], AIS_cal_chest(pred_dmax), [0, 2, 3, 4, 5])
    cls_metrics_neck = get_classification_metrics(ground_truths['ais_neck'], AIS_cal_neck(pred_nij), [0, 2, 3, 4, 5])

    # MAIS 指标
    mais_pred = np.maximum.reduce([AIS_cal_head(pred_hic), AIS_cal_chest(pred_dmax), AIS_cal_neck(pred_nij)])
    cls_metrics_mais = get_classification_metrics(ground_truths['mais'], mais_pred, [0, 1, 2, 3, 4, 5])
    
    # HIC 特有的 AIS-3C 指标
    cls_metrics_hic_3c = get_classification_metrics(AIS_3_cal_head(true_hic), AIS_3_cal_head(pred_hic), [0, 1, 3])

    # --- 4. 生成并保存所有可视化图表 ---
    plot_scatter(true_hic, pred_hic, ground_truths['ais_head'], 'Head Injury Criterion (HIC)', 'HIC', os.path.join(args.run_dir, "scatter_plot_HIC.png"))
    plot_scatter(true_dmax, pred_dmax, ground_truths['ais_chest'], 'Chest Displacement (Dmax)', 'Dmax (mm)', os.path.join(args.run_dir, "scatter_plot_Dmax.png"))
    plot_scatter(true_nij, pred_nij, ground_truths['ais_neck'], 'Neck Injury Criterion (Nij)', 'Nij', os.path.join(args.run_dir, "scatter_plot_Nij.png"))

    plot_confusion_matrix(cls_metrics_head['conf_matrix'], [0, 1, 2, 3, 4, 5], 'Confusion Matrix - AIS Head (6C)', os.path.join(args.run_dir, "cm_head_6c.png"))
    plot_confusion_matrix(cls_metrics_chest['conf_matrix'], [0, 2, 3, 4, 5], 'Confusion Matrix - AIS Chest (5C)', os.path.join(args.run_dir, "cm_chest_5c.png"))
    plot_confusion_matrix(cls_metrics_neck['conf_matrix'], [0, 2, 3, 4, 5], 'Confusion Matrix - AIS Neck (5C)', os.path.join(args.run_dir, "cm_neck_5c.png"))
    plot_confusion_matrix(cls_metrics_mais['conf_matrix'], [0, 1, 2, 3, 4, 5], 'Confusion Matrix - MAIS (6C)', os.path.join(args.run_dir, "cm_mais_6c.png"))
    plot_confusion_matrix(cls_metrics_hic_3c['conf_matrix'], [0, 1, 3], 'Confusion Matrix - AIS Head (3C)', os.path.join(args.run_dir, "cm_head_3c.png"))
    print(f"All plots have been saved to {args.run_dir}")

    # --- 5. 生成并保存 Markdown 报告 ---
    markdown_content = f"""# Model Evaluation Report

## Model Identification
- **Run Directory**: `{args.run_dir}`
- **Weight File**: `{args.weight_file}`
- **Model Type**: {"Teacher" if "teacher" in args.weight_file.lower() else "Student"}
- **Testset size**: {len(test_dataset)}
```

## Overall Injury Assessment (MAIS)

- **Accuracy**: {cls_metrics_mais['accuracy']:.2f}%
- **G-Mean**: {cls_metrics_mais['g_mean']:.4f}
- **Confusion Matrix**:
{cls_metrics_mais['conf_matrix']}
- **Classification Report**:
{cls_metrics_mais['report']}

---
"""
    markdown_content += generate_report_section("Head (HIC)", reg_metrics_hic, cls_metrics_head, cls_metrics_hic_3c)
    markdown_content += "---\n"
    markdown_content += generate_report_section("Chest (Dmax)", reg_metrics_dmax, cls_metrics_chest)
    markdown_content += "---\n"
    markdown_content += generate_report_section("Neck (Nij)", reg_metrics_nij, cls_metrics_neck)

    report_path = os.path.join(args.run_dir, f"TestResults_{args.weight_file.replace('.pth', '')}.md")
    with open(report_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)
    
    print(f"Comprehensive evaluation report saved to {report_path}")