''' This module includes data precessing dataset creating. '''
# 2025.01.04

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# Define the random seed.
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class SigmoidTransform(nn.Module): 
    """
    对标签值HIC进行 Sigmoid 变换和反变换。感兴趣范围被映射到[sigmoid(-2), sigmoid(2)]
    相当于隐性地为不同HIC的样本在训练中加权
    反变换容易数值溢出(比如原始HIC>12000, >10000后就有不小误差)
    """
    def __init__(self, lower_bound, upper_bound):
        """
        初始化 SigmoidTransform。
        
        参数:
            lower_bound (float): 感兴趣范围的下界(如200)。
            upper_bound (float): 感兴趣范围的上界(如2000)。
        """
        super(SigmoidTransform, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.a = 4 / (upper_bound - lower_bound)  # 缩放因子
        self.b = -2 - self.a * lower_bound        # 偏移量

    def forward(self, y):
        """
        对标签值进行 Sigmoid 变换。
        
        参数:
            y (torch.Tensor): 输入标签值，形状为(B,)。
        
        返回:
            torch.Tensor: 变换后的标签值，形状为(B,)。
        """
        return torch.sigmoid(self.a * y + self.b) 

    def inverse(self, y_transformed):
        """
        对 Sigmoid 变换后的标签值进行反变换。
        容易数值溢出(比如原始HIC>12000, >10000后就有不小误差)
        
        参数:
            y_transformed (torch.Tensor): 变换后的标签值，形状为(B,)。
        
        返回:
            torch.Tensor: 反变换后的标签值，形状为(B,)。
        """
        return (torch.log(y_transformed / (1 - y_transformed)) - self.b) / self.a

# 经验公式计算AIS
def AIS_3_cal(HIC):
    HIC = np.clip(HIC, 1, 2500)
    coefficients = np.array([
        [1.54, 0.00650],  # P(AIS≥1)
        [3.39, 0.00372]   # P(AIS≥3)
    ])
    threshold = 0.2
    c1 = coefficients[:, 0].reshape(-1, 1)
    c2 = coefficients[:, 1].reshape(-1, 1)
    HIC_inv = 200 / HIC
    hic_prob = 1.0 / (1.0 + np.exp(c1 + HIC_inv - c2 * HIC))
    AIS_3 = np.sum(hic_prob.T >= threshold, axis=1)
    return AIS_3

def AIS_cal(HIC, prob_output=False):
    # 限制 HIC 范围，防止数值不稳定
    HIC = np.clip(HIC, 1, 2500)

    # 定义常量和系数
    coefficients = np.array([
        [1.54, 0.00650],  # P(AIS≥1)
        [2.49, 0.00483],  # P(AIS≥2)
        [3.39, 0.00372],  # P(AIS≥3)
        [4.90, 0.00351],  # P(AIS≥4)
        [7.82, 0.00429]   # P(AIS≥5)
    ])
    threshold = 0.2  # 经验概率阈值

    # 计算 P(AIS≥n) 的概率（向量化计算）
    c1 = coefficients[:, 0].reshape(-1, 1)  # 系数1
    c2 = coefficients[:, 1].reshape(-1, 1)  # 系数2
    HIC_inv = 200 / HIC  # HIC 的倒数部分

    # 计算所有 P(AIS≥n)
    hic_prob = 1.0 / (1.0 + np.exp(c1 + HIC_inv - c2 * HIC))

    # 计算 P(AIS=n) 的概率
    ais_prob = np.zeros((len(HIC), 6))  # 初始化 (样本数, 6)
    ais_prob[:, 0] = 1 - hic_prob[0]  # P(AIS=0)
    ais_prob[:, 1] = hic_prob[0] - hic_prob[1]  # P(AIS=1)
    ais_prob[:, 2] = hic_prob[1] - hic_prob[2]  # P(AIS=2)
    ais_prob[:, 3] = hic_prob[2] - hic_prob[3]  # P(AIS=3)
    ais_prob[:, 4] = hic_prob[3] - hic_prob[4]  # P(AIS=4)
    ais_prob[:, 5] = hic_prob[4]  # P(AIS=5)

    # 确定 AIS 等级
    AIS = np.sum(hic_prob.T >= threshold, axis=1)

    if prob_output:
        return AIS, ais_prob
    else:
        return AIS
    
class CrashDataset(Dataset):
    def __init__(self, acc_file='./data/data_crashpulse.npy', att_file='./data/data_features.npy', y_transform=None):
        """
        Args:
            acc_file (str): 碰撞波形数据的文件路径 (x_acc)。
            att_file (str): 特征数据的文件路径 (x_att)。
            y_transform (callable, optional): 可选是否对标签HIC值进行变换。默认为 None。
        """
        self.x_acc = np.load(acc_file)  # 加载碰撞波形数据  npdarray (5777, 2, 150)
        self.x_att = np.load(att_file)  # 加载其他特征数据  npdarray (5777, 9)
        self.continuous_indices = [0, 1, 3, 4]  # 连续特征的索引
        self.discrete_indices = [2, 5, 6, 7]  # 离散特征的索引

        # 初始化 LabelEncoder 和映射关系存储
        self.label_encoders = {}  # 存储每个离散特征的 LabelEncoder
        self.encoding_mappings = {}  # 存储每个离散特征的编码映射关系

        # 数据预处理
        self._preprocess_data()

        # 目标变量
        self.y_HIC = self.x_att[:, 8]  # HIC 值
        self.y_AIS = AIS_cal(self.y_HIC)  # 计算 AIS-6C 值
        
        # print(f"y_HIC中<100的值的数量: {len(np.where(self.y_HIC < 100)[0])}")
        # print(f"y_HIC中>2000的值的数量: {len(np.where(self.y_HIC > 2000)[0])}")
        # print(f"y_HIC最大值: {np.max(self.y_HIC)}")

        # 实际输入特征
        self.x_att = self.x_att[:, :8]

        # 存储每个离散特征的类别数
        self.num_classes_of_discrete = [
            len(self.label_encoders[idx].classes_) for idx in self.discrete_indices]

        self.y_transform = y_transform

    def _preprocess_data(self):
        """
        数据预处理, 分别处理连续特征和离散特征
        """
        # 碰撞波形数据 (x_acc)
        # 形状 (5777, 2, 150)，2 表示 X 和 Y 方向，150 表示时间步长
        # 可能存在噪声，不够平滑
        # 对 X 和 Y 方向的波形数据进行归一化
        for i in range(2):  # 分别处理 X 和 Y 方向
            min_val = np.min(self.x_acc[:, i])
            max_val = np.max(self.x_acc[:, i])
            self.x_acc[:, i] = (self.x_acc[:, i] - min_val) / (max_val - min_val)  # 归一化到 [0, 1]

        # 特征数据 (x_att)
        # 形状 (5777, 9)，9 个特征，包括连续和离散变量
        # 特征索引及含义：
        # 0: ego vehicle speed (连续): 23.0~140.0km/h
        # 1: leading vehicle speed (连续): 10.0~120.0km/h
        # 2: collision overlap rate (离散，0~6)
        # 3: collision angle (连续): -30~30°
        # 4: mass of the leading vehicle (连续): 900.0-3900.0kg
        # 5: belt usage (离散，0/1)
        # 6: airbag usage (离散，0/1)
        # 7: occupant size (离散，0/1/2)
        # 8: HIC (连续，目标变量): 0 ~ >2000, 最大的在30000-40000之间

        # 处理连续特征
        for idx in self.continuous_indices:
            min_val = np.min(self.x_att[:, idx])
            max_val = np.max(self.x_att[:, idx])
            self.x_att[:, idx] = (self.x_att[:, idx] - min_val) / (max_val - min_val)  # 归一化到 [0, 1]

        # 处理离散特征
        for idx in self.discrete_indices:
            le = LabelEncoder()
            self.x_att[:, idx] = le.fit_transform(self.x_att[:, idx])  # 转换为从 0 开始的整数
            self.label_encoders[idx] = le  # 存储 LabelEncoder
            self.encoding_mappings[idx] = dict(zip(le.classes_, le.transform(le.classes_)))  # 存储编码映射关系

    def print_encoding_mappings(self):
        """
        打印离散特征的编码映射关系。
        """
        for idx, mapping in self.encoding_mappings.items():
            print(f"Feature {idx} encoding mapping: {mapping}")

    def __len__(self):
        return len(self.x_acc)

    def __getitem__(self, idx):
        """
        根据索引获取数据。
        """
        x_acc = self.x_acc[idx]  # 碰撞波形数据，形状 (2, 150)
        x_att = self.x_att[idx]  # 特征数据，形状 (8,)
        y_HIC = self.y_HIC[idx]  # HIC 值，标量
        y_AIS = self.y_AIS[idx]  # AIS 值，标量

        x_att_continuous = x_att[self.continuous_indices]  # 连续特征，形状 (4,)
        x_att_discrete = x_att[self.discrete_indices]      # 离散特征，形状 (4,)
        
        # 如果提供了标签变换函数，则对标签进行变换
        if self.y_transform is not None:
            y_HIC = self.y_transform(torch.tensor(y_HIC, dtype=torch.float32))

        return (
            torch.tensor(x_acc, dtype=torch.float32),          # 碰撞波形数据，float32
            torch.tensor(x_att_continuous, dtype=torch.float32),  # 连续特征，float32
            torch.tensor(x_att_discrete, dtype=torch.long),       # 离散特征，long
            torch.tensor(y_HIC, dtype=torch.float32),          # HIC 值，float32
            torch.tensor(y_AIS, dtype=torch.long)              # AIS 值，long
        )

if __name__ == '__main__':
    import time
    # TEST
    start_time = time.time()
    HIC_transform = SigmoidTransform(0, 2500)
    dataset = CrashDataset(y_transform=HIC_transform)
    print("Dataset loading time:", time.time() - start_time)

    train_size = 5000
    val_size = 500
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 把三者保存到本地
    if dataset.y_transform is None:
        torch.save(train_dataset, './data/train_dataset.pt')
        torch.save(val_dataset, './data/val_dataset.pt')
        torch.save(test_dataset, './data/test_dataset.pt')
    else:
        torch.save(train_dataset, './data/train_dataset_ytrans.pt')
        torch.save(val_dataset, './data/val_dataset_ytrans.pt')
        torch.save(test_dataset, './data/test_dataset_ytrans.pt')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    dataset.print_encoding_mappings()
    
    batch_start_time = time.time()
    for i, batch in enumerate(train_loader):
        
        x_acc, x_att_continuous, x_att_discrete, y_HIC, y_AIS = batch
        print(x_acc.shape, x_att_continuous.shape, x_att_discrete.shape, y_HIC.shape, y_AIS.shape)
        break
    print("batch time:", time.time() - batch_start_time)
