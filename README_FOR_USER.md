# **乘员损伤预测模型**

本项目包含两种用于预测乘员损伤（头部HIC、胸部Dmax、颈部Nij）的深度学习模型。

* **教师模型 (Teacher Model)**: 一个基于时序卷积网络 (TCN) 和多层感知机 (MLP) 的模型。它使用碰撞波形数据和工况标量数据作为输入，精度较高，适用于详细的离线分析。  
* **学生模型 (Student Model)**: 一个仅基于多层感知机 (MLP) 的轻量级模型。它仅使用工况标量数据作为输入，计算速度快，适用于需要快速推理的部署场景。

## **1\. 项目结构**

.  
├── runs/                 \# 存放所有训练运行的日志、模型权重和结果  
├── data/                 \# 存放处理后的数据集文件 (.pt)  
├── utils/                \# 功能模块与辅助函数  
│   ├── models.py             \# 定义教师和学生模型的网络结构  
│   ├── dataset\_prepare.py    \# 数据集预处理与加载  
│   ├── weighted\_loss.py      \# 自定义加权损失函数  
│   └── AIS\_cal.py            \# AIS等级计算函数  
├── train\_teacher.py      \# 脚本：训练教师模型  
├── train\_student\_w\_KD.py \# 脚本：使用知识蒸馏训练学生模型  
├── train\_student\_wo\_KD.py\# 脚本：独立训练学生模型 (作为基准)  
└── eval\_model.py         \# 脚本：评估一个已训练好的模型的性能

## **2\. 环境设置**

本项目基于 Python 和 PyTorch。请确保您已安装所有必要的依赖。

1. **安装依赖包**:  
   pip install \-r requirements.txt

   所有依赖项已在 requirements.txt 中列出。

## **3\. 使用说明 (推理与评估)**

本节主要面向直接使用已训练好的模型进行推理和性能评估的用户。

### **步骤 1: 准备数据集**

评估脚本 (eval\_model.py) 需要加载测试数据集。如果您的 data/ 目录下还没有 test\_dataset.pt 和 val\_dataset.pt 文件，请运行一次数据预处理脚本来生成它们。

python utils/dataset\_prepare.py

此脚本会利用原始数据生成 train\_dataset.pt, val\_dataset.pt, test\_dataset.pt 文件并存放在 ./data/ 目录下。

### **步骤 2: 执行评估脚本**

使用 eval\_model.py 脚本来加载模型权重并在测试集上进行评估。您需要提供两个关键参数：

* \--run\_dir 或 \-r: 存放模型权重和 TrainingRecord.json 文件的运行目录。  
* \--weight\_file 或 \-w: 要加载的模型权重文件名（.pth 文件）。

**示例:**

* **评估一个教师模型:**  
  python eval\_model.py \-r ./runs/TeacherModel\_Train\_XXXXXXXX \-w teacher\_best\_mais\_accu.pth

* **评估一个学生模型:**  
  python eval\_model.py \-r ./runs/StudentModel\_Distill\_XXXXXXXX \-w student\_best\_mais\_accu.pth

请将 XXXXXXXX 替换为您实际的运行目录的时间戳。

### **步骤 3: 查看结果**

脚本执行完毕后，会生成以下产出：

1. **控制台输出**: 在终端直接打印核心的评估指标。  
2. **详细评估报告**: 在指定的 \--run\_dir 目录下生成一个 Markdown 格式的详细报告 (TestResults\_\*.md)，包含所有损伤部位的回归和分类指标。  
3. **可视化图表**: 在 \--run\_dir 目录下生成并保存以下图片文件：  
   * 三种损伤指标（HIC, Dmax, Nij）的预测值 vs. 真实值散点图。  
   * 四种AIS分类（头部、胸部、颈部、MAIS）的混淆矩阵图。

## **4\. 模型推理接口说明 (如何获取预测结果)**

如果您希望将预训练模型集成到自己的代码中进行推理，请遵循以下步骤：

### **步骤 1: 加载模型**

import torch  
from utils import models

\# 假设您已从 TrainingRecord.json 中加载了模型超参数 model\_params  
\# model \= models.TeacherModel(\*\*model\_params, num\_classes\_of\_discrete=...)  
\# model.load\_state\_dict(torch.load('path/to/your/model.pth'))  
\# model.eval()

### **步骤 2: 准备输入数据**

根据您使用的模型，准备形状正确的 PyTorch 张量。

* **教师模型 forward 函数输入**:  
  1. x\_acc (torch.Tensor): 碰撞波形数据，形状为 (B, 3, 150)。  
  2. x\_att\_continuous (torch.Tensor): 连续标量特征，形状为 (B, 14)。  
  3. x\_att\_discrete (torch.Tensor): 离散标量特征，形状为 (B, 4)。  
* **学生模型 forward 函数输入**:  
  1. x\_att\_continuous (torch.Tensor): 连续标量特征，形状为 (B, 14)。  
  2. x\_att\_discrete (torch.Tensor): 离散标量特征，形状为 (B, 4)。

### **步骤 3: 执行推理并解析输出**

模型的 forward 方法返回一个形状为 (B, 3\) 的张量，按顺序分别对应 HIC, Dmax, Nij 的预测值。您可以使用 utils/AIS\_cal.py 中的函数来获取最终的AIS等级。

import numpy as np  
from utils.AIS\_cal import AIS\_cal\_head, AIS\_cal\_chest, AIS\_cal\_neck

\# 假设 model 和 input\_tensors 已准备好  
\# with torch.no\_grad():  
\#     predictions\_tensor, \_, \_ \= model(\*input\_tensors)

\# 将输出转换为Numpy数组  
\# predictions\_np \= predictions\_tensor.cpu().numpy()

\# 1\. 获取三个部位的损伤标量值  
\# pred\_hic \= predictions\_np\[:, 0\]  
\# pred\_dmax \= predictions\_np\[:, 1\]  
\# pred\_nij \= predictions\_np\[:, 2\]

\# 2\. 计算对应的AIS等级  
\# ais\_head \= AIS\_cal\_head(pred\_hic)  
\# ais\_chest \= AIS\_cal\_chest(pred\_dmax)  
\# ais\_neck \= AIS\_cal\_neck(pred\_nij)

\# 3\. 计算MAIS  
\# mais \= np.maximum.reduce(\[ais\_head, ais\_chest, ais\_neck\])

\# 现在您已获得所有预测结果：  
\# pred\_hic, pred\_dmax, pred\_nij, ais\_head, ais\_chest, ais\_neck, mais  
