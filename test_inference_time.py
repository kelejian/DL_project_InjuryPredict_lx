import os
import json
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from utils import models
from utils.dataset_prepare import CrashDataset, SigmoidTransform

# 定义随机种子
seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_inference_time(model, loader, y_transform=None):
    """
    测试模型推理时间
    参数:
        model: 模型实例。
        loader: 数据加载器。
        y_transform: 数据集中的HIC标签变换对象。若数据集中的HIC标签没有进行变换则为None。
    """
    model.eval()
    total_time = 0.0
    num_runs = 200  # 推理次数

    with torch.no_grad():
        for _ in range(num_runs):
            for batch_x_acc, batch_x_att_continuous, batch_x_att_discrete, batch_y_HIC, batch_y_AIS in loader:
                # 将数据移动到设备
                batch_x_acc = batch_x_acc.to(device)
                batch_x_att_continuous = batch_x_att_continuous.to(device)
                batch_x_att_discrete = batch_x_att_discrete.to(device)
                batch_y_HIC = batch_y_HIC.to(device)

                # 预热
                if _ == 0:
                    for _ in range(50):
                        if isinstance(model, models.TeacherModel):
                            model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)
                        elif isinstance(model, models.StudentModel):
                            model(batch_x_att_continuous, batch_x_att_discrete)

                # 开始计时
                start_time = time.time()

                # 前向传播
                if isinstance(model, models.TeacherModel):
                    model(batch_x_acc, batch_x_att_continuous, batch_x_att_discrete)
                elif isinstance(model, models.StudentModel):
                    model(batch_x_att_continuous, batch_x_att_discrete)

                # 结束计时
                elapsed_time = time.time() - start_time
                total_time += elapsed_time

    # 计算平均推理时间
    avg_time = total_time / num_runs
    print(f"Average inference time: {avg_time:.4f} seconds")

if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="Test Teacher or Student Model Inference Time")
    parser.add_argument("--run_dir", '-r', type=str, default=".\\runs\\StudentModel_Distill_01122148")
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
    # 提取 HIC_transform 参数
    HIC_transform_params = train_params.get("HIC_transform")  # 如果没有这个键，返回 None

    # 初始化 HIC_transform
    if HIC_transform_params is not None:
        HIC_transform = SigmoidTransform(
            lower_bound=HIC_transform_params["lower_bound"],
            upper_bound=HIC_transform_params["upper_bound"]
        )
    else:
        HIC_transform = None  # 如果没有 HIC_transform 参数，设置为 None

    # 加载数据集
    dataset = CrashDataset(y_transform=HIC_transform)
    if dataset.y_transform is not None:
        test_dataset1 = torch.load("./data/val_dataset_ytrans.pt")
        test_dataset2 = torch.load("./data/test_dataset_ytrans.pt")
    else:
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

    # 测试推理时间
    test_inference_time(model, test_loader, y_transform=dataset.y_transform)