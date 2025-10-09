''' This module includes data precessing dataset creating. '''
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np

try:
    from utils.set_random_seed import GLOBAL_SEED, set_random_seed  # 作为包导入时使用
except ImportError:
    from set_random_seed import GLOBAL_SEED, set_random_seed   # 直接运行时使用

set_random_seed()

class CrashDataset(Dataset):
    def __init__(self, input_file='./data/data_input.npz', label_file='./data/data_labels.npz'):
        """
        Args:
            input_file (str): 包含碰撞波形和特征数据的 .npz 文件路径。
            label_file (str): 包含标签数据的 .npz 文件路径。
        """
        self.inputs = np.load(input_file)
        self.labels = np.load(label_file)

        # --- 新增：对齐校验 ---
        inp_ids = self.inputs['case_ids']
        lab_ids = self.labels['case_ids']
        assert np.array_equal(inp_ids, lab_ids), (
            f"Case ID 不匹配：input_file 中 {inp_ids[:5]}… vs label_file 中 {lab_ids[:5]}…"
        )

        # 只有校验通过才继续下面的赋值和预处理
        self.case_ids = inp_ids # 形状 (N,)

        # 输入特征
        self.x_acc = self.inputs['waveforms'] # 形状 (N, 3, 150) acceleration waveforms
        self.x_att = self.inputs['params'] # 形状 (N, 18)  attributes

        # --- 加载所有目标变量 ---
        self.y_HIC = self.labels['HIC']
        self.y_Dmax = self.labels['Dmax']
        self.y_Nij = self.labels['Nij']
        self.ais_head = self.labels['AIS_head']
        self.ais_chest = self.labels['AIS_chest']
        self.ais_neck = self.labels['AIS_neck']
        self.mais = self.labels['MAIS']

        # 初始化 LabelEncoder 和映射关系存储
        self.label_encoders = {}  # 存储每个离散特征的 LabelEncoder
        self.encoding_mappings = {}  # 存储每个离散特征的编码映射关系

        # 数据预处理
        self._preprocess_data()
        
        # 存储每个离散特征的类别数
        self.num_classes_of_discrete = [
            len(self.label_encoders[idx].classes_) for idx in self.discrete_indices]

        # 关闭 npz 文件以避免 pickle 错误
        self.inputs.close()
        self.labels.close()

    def _preprocess_data(self):
        """
        数据预处理, 分别处理连续特征和离散特征
        """
        # 碰撞波形数据 (x_acc)
        # 形状 (N, 3, 150)，3 表示 X 和 Y 和 Z 方向，150 表示时间步长
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

        self.continuous_indices = [0, 1, 2, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 17]  # 连续特征的索引
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
        x_acc = self.x_acc[idx]
        x_att = self.x_att[idx]
        
        # --- 提取所有标签 ---
        y_HIC = self.y_HIC[idx]
        y_Dmax = self.y_Dmax[idx]
        y_Nij = self.y_Nij[idx]
        ais_head = self.ais_head[idx]
        ais_chest = self.ais_chest[idx]
        ais_neck = self.ais_neck[idx]
        mais = self.mais[idx]

        x_att_continuous = x_att[self.continuous_indices]
        x_att_discrete = x_att[self.discrete_indices]
        
        return (
            torch.tensor(x_acc, dtype=torch.float32),
            torch.tensor(x_att_continuous, dtype=torch.float32),
            torch.tensor(x_att_discrete, dtype=torch.long),
            # --- 返回所有损伤指标和AIS等级 ---
            torch.tensor(y_HIC, dtype=torch.float32),
            torch.tensor(y_Dmax, dtype=torch.float32),
            torch.tensor(y_Nij, dtype=torch.float32),
            torch.tensor(ais_head, dtype=torch.long),
            torch.tensor(ais_chest, dtype=torch.long),
            torch.tensor(ais_neck, dtype=torch.long),
            torch.tensor(mais, dtype=torch.long)
        )

if __name__ == '__main__':
    # 导入所需的库
    import time
    import numpy as np
    import pandas as pd
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split

    # 分割打包数据集并测试
    start_time = time.time()
    dataset = CrashDataset()
    print("\nDataset loading time:", time.time() - start_time)

    # --- Robust Stratified Splitting ---

    # 1. 定义数据集划分比例
    # 即使测试集比例很小，也在这里定义，方便未来调整
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.19
    # TEST_RATIO 将是剩余部分，确保总和为1
    TEST_RATIO = round(1.0 - TRAIN_RATIO - VAL_RATIO, 2)
    if TEST_RATIO < 0:
        raise ValueError("TRAIN_RATIO and VAL_RATIO cannot sum to more than 1.")


    # 2. 准备用于分层的标签和索引
    labels = dataset.mais
    indices = np.arange(len(dataset))

    # 3. 第一次划分：严格分层地划分出训练集 vs. (验证集 + 测试集)
    
    # 找出并分离出样本数少于2的 "孤儿" 类别，直接放入训练集
    label_counts = pd.Series(labels).value_counts()
    # 最小分组数至少为2，所以任何少于2个样本的类别都无法分层
    insufficient_samples_labels = label_counts[label_counts < 2].index.tolist()
    
    train_indices_final = []
    # 剩余待划分的索引
    remaining_indices = indices

    if insufficient_samples_labels:
        print(f"\nWarning: Found classes with < 2 samples: {insufficient_samples_labels}")
        print("These will be moved to the training set to allow stratification.")
        
        # 将孤儿样本的索引直接加入最终的训练集
        singleton_mask = np.isin(labels, insufficient_samples_labels)
        train_indices_final.extend(indices[singleton_mask])
        
        # 从待划分的索引中移除这些孤儿样本
        remaining_indices = indices[~singleton_mask]
    
    remaining_labels = labels[remaining_indices]

    # 对剩余的主体数据进行分层抽样
    # 计算剩余数据中应该有多少比例作为测试/验证集
    temp_size = VAL_RATIO + TEST_RATIO
    
    # Stratify split the rest of the data
    train_main_indices, temp_indices, _, _ = train_test_split(
        remaining_indices, remaining_labels,
        test_size=temp_size,
        random_state=GLOBAL_SEED,
        stratify=remaining_labels
    )
    
    # 将分层抽样出的训练索引与之前的孤儿索引合并
    train_indices_final.extend(train_main_indices)
    train_indices = np.array(train_indices_final)


    # 4. 第二次划分：对 temp_indices 进行 *非分层* 的随机划分
    # 这是关键改动：由于 temp_indices 样本量小，不进行分层以避免错误
    if len(temp_indices) > 0 and TEST_RATIO > 0:
        # 计算验证集和测试集在 temp_indices 中的相对比例
        relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        
        # 检查是否因为样本太少导致无法划分
        if len(temp_indices) < 2:
             # 如果临时集只有一个样本，直接全部分给验证集，测试集为空
             val_indices = temp_indices
             test_indices = []
             print("\nWarning: Not enough samples for a separate test set. Test set will be empty.")
        else:
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=relative_test_ratio,
                random_state=GLOBAL_SEED
                # 注意：此处没有 stratify 参数
            )
    else:
        # 如果 temp_indices 为空或 TEST_RATIO 为0
        val_indices = temp_indices
        test_indices = []

    # 5. 使用 PyTorch 的 Subset 创建最终的数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # --- Splitting End ---

    print(f"\nDataset split sizes:")
    print(f"  - Total: {len(dataset)}")
    print(f"  - Train: {len(train_dataset)}")
    print(f"  - Validation: {len(val_dataset)}")
    print(f"  - Test: {len(test_dataset)}")

    # 验证标签分布
    def get_label_distribution(subset):
        if len(subset.indices) == 0:
            return "Empty"
        labels = [subset.dataset.mais[i] for i in subset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

    print("\nMAIS label distribution in each subset:")
    print(f"  - Train: {get_label_distribution(train_dataset)}")
    print(f"  - Validation: {get_label_distribution(val_dataset)}")
    print(f"  - Test: {get_label_distribution(test_dataset)}")

    # 保存处理后的数据集
    torch.save(train_dataset, './data/train_dataset.pt')
    torch.save(val_dataset, './data/val_dataset.pt')
    torch.save(test_dataset, './data/test_dataset.pt')
    print("\nTrain, validation, and test datasets saved successfully.")


    # 测试 DataLoader 是否能正常工作
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    dataset.print_encoding_mappings()
    
    print("\nTesting DataLoader...")
    batch_start_time = time.time()
    for i, batch in enumerate(train_loader):
        (x_acc, x_att_continuous, x_att_discrete, 
         y_HIC, y_Dmax, y_Nij, 
         ais_head, ais_chest, ais_neck, mais) = batch
        
        print("x_acc shape:", x_acc.shape)
        print("y_HIC shape:", y_HIC.shape)
        print("MAIS shape:", mais.shape)
        break
    print("batch loading time:", time.time() - batch_start_time)