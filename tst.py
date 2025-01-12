
# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the function
# Define the function
_HIC = np.linspace(0, 2500, 2000)
hic = np.zeros((5, len(_HIC)))
ais_prob = np.zeros((6, len(_HIC)))

hic[0] = 1. / (1 + np.exp(1.54 + 200 / _HIC - 0.00650 * _HIC))  # P(AIS≥1)
ais_prob[0] = 1 - hic[0]  # AIS=0的概率
hic[1] = 1. / (1 + np.exp(2.49 + 200 / _HIC - 0.00483 * _HIC))  # P(AIS≥2)
ais_prob[1] = hic[0] - hic[1]  # AIS=1的概率
hic[2] = 1. / (1 + np.exp(3.39 + 200 / _HIC - 0.00372 * _HIC))  # P(AIS≥3)
ais_prob[2] = hic[1] - hic[2]  # AIS=2的概率
hic[3] = 1. / (1 + np.exp(4.90 + 200 / _HIC - 0.00351 * _HIC))  # P(AIS≥4)
ais_prob[3] = hic[2] - hic[3]  # AIS=3的概率
hic[4] = 1. / (1 + np.exp(7.82 + 200 / _HIC - 0.00429 * _HIC))  # P(AIS≥5)
ais_prob[4] = hic[3] - hic[4]  # AIS=4的概率
ais_prob[5] = hic[4]  # AIS=5的概率

# 在一张图中绘制 AIS 各个概率随 _HIC 变化的曲线
plt.figure(figsize=(10, 6))
plt.plot(_HIC, hic[0], label="P(AIS>=1|HIC)", color="blue")
plt.plot(_HIC, hic[1], label="P(AIS>=2|HIC)", color="green")
plt.plot(_HIC, hic[2], label="P(AIS>=3|HIC)", color="red")
plt.plot(_HIC, hic[3], label="P(AIS>=4|HIC)", color="purple")
plt.plot(_HIC, hic[4], label="P(AIS>=5|HIC)", color="orange")
plt.xlabel("HIC", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.title("AIS Probability vs. HIC", fontsize=14)
plt.grid()
plt.legend(fontsize=10)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
T_max = 50  # 周期总步长
eta_max = 0.1  # 初始最大学习率
eta_min = 0.01  # 最小学习率

# 定义 lr(t) 数学表达式
def cosine_annealing_lr(t, T_max, eta_max, eta_min):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T_max))

# 生成 t 和 lr(t) 数据
t_values = np.linspace(0, 2*T_max, 1000)  # 模拟 t 从 0 到 T_max
lr_values = cosine_annealing_lr(t_values, T_max, eta_max, eta_min)

# 绘制曲线
plt.plot(t_values, lr_values, label="CosineAnnealingLR", color="blue")
plt.xlabel("t (当前步数)", fontsize=12)
plt.ylabel("Learning Rate (学习率)", fontsize=12)
plt.title("CosineAnnealingLR 数学表达式曲线", fontsize=14)
plt.grid()
plt.legend(fontsize=10)
plt.show()

# %%
import torch

# 定义输入
pred_ais_probs = torch.tensor([
    [0.1, 0.6, 0.2, 0.05, 0.025, 0.025],  # 样本1的预测概率分布
    [0.7, 0.1, 0.1, 0.05, 0.05, 0.0],    # 样本2的预测概率分布
    [0.05, 0.15, 0.4, 0.3, 0.05, 0.05],  # 样本3的预测概率分布
    [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]       # 样本4的预测概率分布
], dtype=torch.float32)  # [batch_size=4, num_classes=6]

true_ais = torch.tensor([1, 0, 2, 3], dtype=torch.long)  # 样本的真实类别标签

# 定义函数
def cross_entropy_loss_from_prob(pred_ais_probs, true_ais):
    """
    计算基于经验概率的交叉熵损失。
    Args:
        pred_ais_probs (torch.Tensor): 经验概率，形状为 [batch_size, num_classes]。
        true_ais (torch.Tensor): 实际的类别标签，形状为 [batch_size]。
    Returns:
        torch.Tensor: 交叉熵损失标量。
    """
    # 获取每个样本真实类别对应的概率
    true_class_probs = pred_ais_probs[torch.arange(true_ais.size(0)), true_ais]

    # 计算交叉熵损失
    loss = -torch.log(true_class_probs + 1e-25).mean()  # 加 1e-8 防止 log(0) 数值溢出

    return loss

# 计算损失
loss = cross_entropy_loss_from_prob(pred_ais_probs, true_ais)
print(loss.shape)

print((torch.log(torch.tensor([1e-25]))).shape)




# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def swish(x):
    return x / (1 + np.exp(-x))

def softplus(x):
    return np.log(1 + np.exp(x))

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def relu6(x):
    return np.minimum(np.maximum(0, x), 6)
x = np.linspace(-10, 10, 500)
# Calculate the values for the functions
swish_vals = swish(x)
softplus_vals = softplus(x)
mish_vals = mish(x)
relu6_vals = relu6(x)
sigmoid = 1 / (1 + np.exp(-x))

# Plot all functions
plt.figure(figsize=(10, 8))
plt.plot(x, sigmoid, label='Sigmoid', linestyle='--', linewidth=2)
plt.plot(x, swish_vals, label='Swish', linewidth=2)
plt.plot(x, softplus_vals, label='Softplus', linewidth=2)
plt.plot(x, mish_vals, label='Mish', linewidth=2)
plt.plot(x, relu6_vals, label='ReLU6', linewidth=2)
plt.axhline(0, color='black', linewidth=0.5, linestyle='dotted')
plt.axvline(0, color='black', linewidth=0.5, linestyle='dotted')
plt.title("Comparison of Activation Functions", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def weighted_function_alpha(x, threshold=1000, k1=1, k2=1, k3=10):
    alpha = 1 / (1 + np.exp(-k3 * (x - threshold)))
    tanh_part = 0.5 * (np.tanh(k1 * (x - threshold)) + 1)
    sigmoid_part = 1 / (1 + np.exp(-k2 * (x - threshold)))
    return alpha * sigmoid_part + (1 - alpha) * tanh_part

# Define the weighted function
def weighted_function(x, threshold=1000, ktanh=0.01, ksigmoid=0.004):
    tanh_part = 0.5 * (np.tanh(ktanh * (x - threshold)) + 1)
    sigmoid_part = 1 / (1 + np.exp(-ksigmoid * (x - threshold)))
    return np.where(x <= threshold, tanh_part, sigmoid_part)

# Generate x values
x = np.linspace(0, 2500, 1000)
# 分类loss在HIC分布上的权重函数
y_class = weighted_function(x, threshold=1000, ktanh=0.01, ksigmoid=0.004)
# MSEloss在HIC分布上的权重函数
y_mse = 1 - y_class
# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y_class,label="y_class")
plt.plot(x, y_mse, label="y_mse")
plt.title("Weighted Function Curve")
plt.xlabel("HIC")
plt.ylabel("w(HIC)")
plt.axvline(1000, color='r', linestyle='--', label="Threshold")
plt.legend()
plt.grid()
plt.show()

# %%
import torch
# Define the simpler weighted function
def weighted_function_simple(x, threshold=1000, ktanh=1, ksigmoid=1):
    tanh_part = 0.5 * (torch.tanh(ktanh * (x - threshold)) + 1)
    sigmoid_part = torch.sigmoid(ksigmoid * (x - threshold))
    return torch.where(x <= threshold, tanh_part, sigmoid_part)

# Example tensor (requires_grad=True to track gradients)
x = torch.linspace(0, 2500, 500, requires_grad=True)
y = weighted_function_simple(x)

# Backward pass (computing gradients)
y.sum().backward()  # Sum up y and compute gradients
print(x.grad)  # Check gradients at each point
# %%
import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x, k):
    return -1 + np.exp(-k * (x - 2500))

# 定义x的范围
x_values = np.linspace(2500, 50000, 1000)

# 定义不同的k值
k_values = [1e-5, 5e-5, 6e-5, 1e-4, 5e-4]

# 绘制图像
plt.figure(figsize=(10, 6))
for k in k_values:
    y_values = f(x_values, k)
    plt.plot(x_values, y_values, label=f'k = {k}')

# 添加标题和标签
plt.title('Function f(x) = -1 + exp(-k(x - 2500)) for different k values')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(y=-1, color='gray', linestyle='--', label='Asymptote y = -1')
plt.axvline(x=2500, color='red', linestyle='--', label='x = 2500')
plt.legend()
plt.grid(True)
plt.show()
# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

def Piecewise_linear(y_true, y_pred, weight_add_mid=1.0):
    """
    计算分段线性权重增加量。

    参数:
        y_true (torch.Tensor): 真实标签，形状为 (B,)。
        y_pred (torch.Tensor): 预测值，形状为 (B,)。
        weight_add_mid (float): 中间区间的权重增加量。

    返回:
        weight_adds (torch.Tensor): 权重增加量，形状与 y_true 相同。
    """
    # 初始化权重增加量为 0
    weight_adds = torch.zeros_like(y_true)
    # 区间 1: 0 <= y <= 150，线性递增至weight_add_mid
    mask = (y_true >= 0) & (y_true < 150)
    weight_adds[mask] = (weight_add_mid / 150) * (y_true[mask] - 0)
    # 区间 2: 150 <= y <= 1600，权重增加量为 weight_add_mid
    mask = (y_true >= 150) & (y_true <= 1600)
    weight_adds[mask] = weight_add_mid
    # 区间 3: 1600 < y < 2000，线性递减，斜率为 -weight_add_mid / 400
    mask = (y_true > 1600) & (y_true < 2000)
    weight_adds[mask] = weight_add_mid + (-weight_add_mid / 400) * (y_true[mask] - 1600)
    # 区间 4: 2000 <= y <= 2500，权重增加量为 0
    mask = (y_true >= 2000) & (y_true <= 2500)
    weight_adds[mask] = 0
    # 区间 5: y > 2500，权重按指数减少
    mask = y_true > 2500
    weight_adds[mask] = -1 + torch.exp(-1e-4 * (y_true[mask] - 2500))

    # 最后增加y_pred<0的权重惩罚
    mask_pred_neg = y_pred < 0
    #weight_adds[mask_pred_neg] += weight_add_mid

    return weight_adds# 生成 y_true 和 y_pred 的值
y_true = torch.linspace(0, 10000, 3000)
y_pred = torch.linspace(-20, 10000, 3000)

# 计算权重增加量
weight_adds = Piecewise_linear(y_true, y_pred)

# 绘制图像
plt.figure(figsize=(12, 6))
plt.plot(y_true.numpy(), weight_adds.numpy(), label='Piecewise Linear Function')
plt.title('Piecewise Linear Function')
plt.xlabel('y_true')
plt.ylabel('Weight Adds')
plt.axhline(y=0, color='gray', linestyle='--')
plt.axvline(x=2500, color='red', linestyle='--', label='x = 2500')
plt.legend()
plt.grid(True)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
x_acc = np.load('./data/data_crashpulse.npy')  # 形状为 (N=5777, 2, T=150)
print(x_acc[0,0,50])
# 提取 X 和 Y 方向的数据
for i in range(2):  # 分别处理 X 和 Y 方向
    min_val = np.min(x_acc[:, i])
    max_val = np.max(x_acc[:, i])
    x_acc[:, i] = (x_acc[:, i] - min_val) / (max_val - min_val)  # 归一化到 [0, 1]
x_direction = x_acc[:, 0, :]  # X 方向数据，形状为 (5777, 150)
y_direction = x_acc[:, 1, :]  # Y 方向数据，形状为 (5777, 150)
num_curves = 5  # 例如绘制 5 条曲线

# 绘制 X 方向的曲线
plt.figure(figsize=(10, 6))
for i in range(num_curves):
    plt.plot(x_direction[i], label=f'X Direction Curve {i+1}')
plt.title('X Direction Curves')
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# 绘制 Y 方向的曲线
plt.figure(figsize=(10, 6))
for i in range(num_curves):
    plt.plot(y_direction[i], label=f'Y Direction Curve {i+1}')
plt.title('Y Direction Curves')
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrix data
confusion_matrix = np.array([[358,  17,   0,   0,   0,   0],
 [ 20,  62,  17,   0,   0,   0],
 [  0,  23,  61,   7,   0,   1],
 [  0,   1,  15,  50,  11,   0],
 [  0,   0,   0,   6,  31,   2],
 [  0,   0,   0,   0,  14,  81]])

# Class labels
labels = ['AIS=0', 'AIS=1', 'AIS=2', 'AIS=3', 'AIS=4', 'AIS=5']

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
# 调大字体
sns.set(font_scale=1.5)
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

# Add labels, title, and axis formatting
#plt.title('Confusion Matrix', fontsize=20)
plt.xlabel('AIS Preds', fontsize=20)
plt.ylabel('AIS Trues', fontsize=20)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# Show the plot
plt.tight_layout()
plt.show()

# %%
