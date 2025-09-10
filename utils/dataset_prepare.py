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
    def __init__(self, input_file='./data/data_input.npz', label_file='./data/hic15_labels.npz', y_transform=None):
        """
        Args:
            input_file (str): 包含碰撞波形和特征数据的 .npz 文件路径。
            label_file (str): 包含标签数据的 .npz 文件路径。
            y_transform (callable, optional): 可选是否对标签HIC值进行变换。默认为 None。
        """
        self.inputs = np.load(input_file)
        self.labels = np.load(label_file)

        self.case_ids = self.inputs['case_ids'] # 形状 (N,)

        # 输入特征
        self.x_acc = self.inputs['waveforms'] # 形状 (N, 3, 150) acceleration waveforms
        self.x_att = self.inputs['params'] # 形状 (N, 18)  attributes

        # 目标变量
        self.y_HIC = self.labels['hic15']  # 形状 (N,)
        self.y_AIS = self.labels['ais']  # 形状 (N,)

        # 初始化 LabelEncoder 和映射关系存储
        self.label_encoders = {}  # 存储每个离散特征的 LabelEncoder
        self.encoding_mappings = {}  # 存储每个离散特征的编码映射关系

        # 数据预处理
        self._preprocess_data()
      
        # print(f"y_HIC中<100的值的数量: {len(np.where(self.y_HIC < 100)[0])}")
        # print(f"y_HIC中>2000的值的数量: {len(np.where(self.y_HIC > 2000)[0])}")
        # print(f"y_HIC最大值: {np.max(self.y_HIC)}")

        # 存储每个离散特征的类别数
        self.num_classes_of_discrete = [
            len(self.label_encoders[idx].classes_) for idx in self.discrete_indices]

        self.y_transform = y_transform

        # 关闭 npz 文件以避免 pickle 错误
        self.inputs.close()
        self.labels.close()

    def _preprocess_data(self):
        """
        数据预处理, 分别处理连续特征和离散特征
        """
        # 碰撞波形数据 (x_acc)
        # 形状 (5777, 3, 150)，3 表示 X 和 Y 和 Z 方向，150 表示时间步长
        # 可能存在噪声，不够平滑，尤其是y和z方向
        # 对 X 和 Y 和 Z 方向的波形数据进行归一化
        for i in range(3):  # 分别处理 X 和 Y 和 Z 方向
            max_val = np.max(self.x_acc[:, i])
            self.x_acc[:, i] = self.x_acc[:, i] / max_val

        # 特征数据 (x_att)
        # 形状 (N, 18)，18 个特征，包括连续和离散变量
        # 特征索引：
        # 0: impact_velocity (连续)
        # 1: impact_angle (连续) 有正负
        # 2: overlap (连续) 有正负
        # 3: occupant_type (离散)
        # 4: ll1 (连续)
        # 5: ll2 (连续)
        # 6: btf (连续)
        # 7: pp (连续)
        # 8: plp (连续)
        # 9: lla_status (离散)
        # 10: llattf (连续)
        # 11: dz (离散)
        # 12: ptf (连续)
        # 13: aft (连续)
        # 14: aav_status (离散)
        # 15: ttf (连续)
        # 16: sp (连续) : 座椅前后位置 (SP - Seat Position) 有正负
        # 17: recline_angle (连续)  座椅靠背角度 (Recline angle) 有正负

        self.continuous_indices = [0, 1, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 17]  # 连续特征的索引
        self.discrete_indices = [3, 9, 11, 14]  # 离散特征的索引

        # 处理连续特征
        for idx in self.continuous_indices:
            min_val = np.min(self.x_att[:, idx])
            max_val = np.max(self.x_att[:, idx])
            if idx in [1, 2, 16, 17]:
                # 归一化考虑正负, 归一化到 [-1, 1]
                self.x_att[:, idx] = self.x_att[:, idx] / max_val
            else:
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
        x_acc = self.x_acc[idx]  # 碰撞波形数据，形状 (3, 150)
        x_att = self.x_att[idx]  # 特征数据，形状 (18,)
        y_HIC = self.y_HIC[idx]  # HIC 值，标量
        y_AIS = self.y_AIS[idx]  # AIS 值，标量

        x_att_continuous = x_att[self.continuous_indices]  # 连续特征，形状 (12,)
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
    #HIC_transform = SigmoidTransform(0, 2500)
    dataset = CrashDataset(y_transform=None)
    print("Dataset loading time:", time.time() - start_time)

    train_size = 500
    val_size = 100
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
