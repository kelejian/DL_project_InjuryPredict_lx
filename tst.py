
# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the function
# Define the function
_HIC = np.linspace(1, 2000, 1000)
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
plt.plot(_HIC, ais_prob[0], label="AIS=0", color="blue")
plt.plot(_HIC, ais_prob[1], label="AIS=1", color="green")
plt.plot(_HIC, ais_prob[2], label="AIS=2", color="red")
plt.plot(_HIC, ais_prob[3], label="AIS=3", color="purple")
plt.plot(_HIC, ais_prob[4], label="AIS=4", color="orange")
plt.plot(_HIC, ais_prob[5], label="AIS=5", color="yellow")
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
