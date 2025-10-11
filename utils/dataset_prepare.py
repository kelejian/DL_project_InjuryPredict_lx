''' This module includes data precessing dataset creating. '''
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import time

try:
    from utils.set_random_seed import GLOBAL_SEED, set_random_seed  # 作为包导入时使用
except ImportError:
    from set_random_seed import GLOBAL_SEED, set_random_seed   # 直接运行时使用

set_random_seed()

class CrashDataset(Dataset):
    """
    数据集类，负责加载和存储原始及处理后的碰撞数据。
    """
    def __init__(self, input_file='./data/data_input.npz', label_file='./data/data_labels.npz'):
        """
        Args:
            input_file (str): 包含碰撞波形和特征数据的 .npz 文件路径。
            label_file (str): 包含标签数据的 .npz 文件路径。
        """
        with np.load(input_file) as inputs, np.load(label_file) as labels:
            # --- 对齐校验 ---
            inp_ids = inputs['case_ids']
            lab_ids = labels['case_ids']
            assert np.array_equal(inp_ids, lab_ids), (
                f"Case ID 不匹配：input_file 中 {inp_ids[:5]}… vs label_file 中 {lab_ids[:5]}…"
            )

            self.case_ids = inp_ids

            # --- 加载原始数据 ---
            self.x_acc_raw = inputs['waveforms'] # 形状 (N, 3, 150) x/y/z direction acceleration waveforms
            self.x_att_raw = inputs['params'] # 形状 (N, 18)  attributes

            # 特征数据 (x_att_raw) 说明：形状 (N, 18)，18 个特征，包括连续和离散变量
            # 0: impact_velocity, 1: impact_angle, 2: overlap, 3: occupant_type, 4: ll1, 5: ll2, 6: btf, 
            # 7: pp, 8: plp, 9: lla_status, 10: llattf, 11: dz, 12: ptf, 13: aft, 14: aav_status, 
            # 15: ttf, 16: sp (座椅前后位置), 17: recline_angle (座椅靠背角度)

            # --- 加载所有目标变量 ---
            self.y_HIC = labels['HIC']
            self.y_Dmax = labels['Dmax']
            self.y_Nij = labels['Nij']
            self.ais_head = labels['AIS_head']
            self.ais_chest = labels['AIS_chest']
            self.ais_neck = labels['AIS_neck']
            self.mais = labels['MAIS']
        
        self.x_acc = None
        self.x_att_continuous = None
        self.x_att_discrete = None

        self.continuous_indices = [0, 1, 2, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 17]
        self.discrete_indices = [3, 9, 11, 14]
        self.num_classes_of_discrete = None

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        if self.x_acc is None or self.x_att_continuous is None or self.x_att_discrete is None:
            raise RuntimeError("数据集尚未预处理。请先运行数据处理流程。")

        return (
            torch.tensor(self.x_acc[idx], dtype=torch.float32),
            torch.tensor(self.x_att_continuous[idx], dtype=torch.float32),
            torch.tensor(self.x_att_discrete[idx], dtype=torch.int),
            torch.tensor(self.y_HIC[idx], dtype=torch.float32),
            torch.tensor(self.y_Dmax[idx], dtype=torch.float32),
            torch.tensor(self.y_Nij[idx], dtype=torch.float32),
            torch.tensor(self.ais_head[idx], dtype=torch.int),
            torch.tensor(self.ais_chest[idx], dtype=torch.int),
            torch.tensor(self.ais_neck[idx], dtype=torch.int),
            torch.tensor(self.mais[idx], dtype=torch.int)
        )

class DataProcessor:
    """
    一个封装了数据预处理逻辑的类，包括拟合(fit)、转换(transform)和结果展示。
    """
    def __init__(self, top_k_waveform=20):
        self.waveform_norm_factor = None
        self.top_k_waveform = top_k_waveform
        self.scaler_minmax = None
        self.scaler_maxabs = None
        self.encoders_discrete = None
        
        self.continuous_indices = [0, 1, 2, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 17]
        self.discrete_indices = [3, 9, 11, 14]
        
        self.minmax_indices_in_continuous = [i for i, orig_idx in enumerate(self.continuous_indices) if orig_idx not in [1, 2, 16, 17]]
        self.maxabs_indices_in_continuous = [i for i, orig_idx in enumerate(self.continuous_indices) if orig_idx in [1, 2, 16, 17]]
        
        self.feature_names = {
            0: "impact_velocity", 1: "impact_angle", 2: "overlap", 3: "occupant_type", 4: "ll1",
            5: "ll2", 6: "btf", 7: "pp", 8: "plp", 9: "lla_status", 10: "llattf", 11: "dz",
            12: "ptf", 13: "aft", 14: "aav_status", 15: "ttf", 16: "sp", 17: "recline_angle"
        }

    def fit(self, train_indices, dataset):
        """
        仅使用训练集数据来拟合所有的scalers和encoders。
        """
        # --- 拟合波形数据的全局归一化因子 ---
        train_x_acc_raw = dataset.x_acc_raw[train_indices]
        # 展平所有波形数据并取绝对值
        flat_abs_waveforms = np.abs(train_x_acc_raw).flatten()
        # 排序并取top k
        top_k_values = np.sort(flat_abs_waveforms)[-self.top_k_waveform:]
        # 计算平均值作为归一化因子
        self.waveform_norm_factor = np.mean(top_k_values)
        if self.waveform_norm_factor < 1e-6: self.waveform_norm_factor = 1.0

        # --- 拟合标量特征的Scaler和Encoder ---
        train_x_att_continuous_raw = dataset.x_att_raw[train_indices][:, self.continuous_indices]
        train_x_att_discrete_raw = dataset.x_att_raw[train_indices][:, self.discrete_indices]

        self.scaler_minmax = MinMaxScaler(feature_range=(0, 1))
        self.scaler_maxabs = MaxAbsScaler()
        self.scaler_minmax.fit(train_x_att_continuous_raw[:, self.minmax_indices_in_continuous])
        self.scaler_maxabs.fit(train_x_att_continuous_raw[:, self.maxabs_indices_in_continuous])
        
        self.encoders_discrete = [LabelEncoder() for _ in range(train_x_att_discrete_raw.shape[1])]
        for i in range(train_x_att_discrete_raw.shape[1]):
            self.encoders_discrete[i].fit(train_x_att_discrete_raw[:, i])

    def transform(self, dataset):
        """
        使用已拟合的处理器转换整个数据集，并填充回dataset对象。
        """
        if self.waveform_norm_factor is None or self.scaler_minmax is None or self.encoders_discrete is None:
            raise RuntimeError("处理器尚未拟合。请先调用fit方法。")

        # --- 转换波形数据 ---
        dataset.x_acc = dataset.x_acc_raw / self.waveform_norm_factor
        
        # --- 转换标量数据 ---
        x_att_continuous_raw = dataset.x_att_raw[:, self.continuous_indices]
        x_att_discrete_raw = dataset.x_att_raw[:, self.discrete_indices]

        x_att_continuous_processed = np.zeros_like(x_att_continuous_raw, dtype=np.float32)
        x_att_continuous_processed[:, self.minmax_indices_in_continuous] = self.scaler_minmax.transform(x_att_continuous_raw[:, self.minmax_indices_in_continuous])
        x_att_continuous_processed[:, self.maxabs_indices_in_continuous] = self.scaler_maxabs.transform(x_att_continuous_raw[:, self.maxabs_indices_in_continuous])
        dataset.x_att_continuous = x_att_continuous_processed

        x_att_discrete_processed = np.zeros_like(x_att_discrete_raw, dtype=np.int64)
        num_classes = []
        for i in range(x_att_discrete_raw.shape[1]):
            x_att_discrete_processed[:, i] = self.encoders_discrete[i].transform(x_att_discrete_raw[:, i])
            num_classes.append(len(self.encoders_discrete[i].classes_))
        dataset.x_att_discrete = x_att_discrete_processed
        dataset.num_classes_of_discrete = num_classes
        
        return dataset
    
    def print_fit_summary(self):
        """
        打印已拟合的scalers和encoders的统计信息。
        """
        if self.waveform_norm_factor is None or self.scaler_minmax is None:
            print("处理器尚未拟合。")
            return
        
        print("\n--- 数据处理器拟合结果摘要 ---")
        
        print(f"\n[碰撞波形 (x_acc) 全局归一化因子]")
        print(f"  - 基于训练集Top {self.top_k_waveform} 最大绝对值点的平均值: {self.waveform_norm_factor:.4f}")

        print("\n[连续标量特征 (x_att_continuous) Scaler 统计量]")
        print("  - MinMaxScaler (归一化至 [0, 1]):")
        for i, idx_in_cont in enumerate(self.minmax_indices_in_continuous):
            orig_idx = self.continuous_indices[idx_in_cont]
            name = self.feature_names.get(orig_idx, f"特征 {orig_idx}")
            print(f"    - {name} (Idx {orig_idx}): Min={self.scaler_minmax.data_min_[i]:.4f}, Max={self.scaler_minmax.data_max_[i]:.4f}")
        
        print("  - MaxAbsScaler (归一化至 [-1, 1]):")
        for i, idx_in_cont in enumerate(self.maxabs_indices_in_continuous):
            orig_idx = self.continuous_indices[idx_in_cont]
            name = self.feature_names.get(orig_idx, f"特征 {orig_idx}")
            print(f"    - {name} (Idx {orig_idx}): MaxAbs={self.scaler_maxabs.max_abs_[i]:.4f}")

        print("\n[离散标量特征 (x_att_discrete) LabelEncoder 映射]")
        for i, encoder in enumerate(self.encoders_discrete):
            orig_idx = self.discrete_indices[i]
            name = self.feature_names.get(orig_idx, f"特征 {orig_idx}")
            mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            print(f"  - {name} (Idx {orig_idx}):")
            print(f"    - 映射关系: {mapping}")
        print("---------------------------------\n")

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

def split_data(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    labels = dataset.mais
    indices = np.arange(len(dataset))
    
    label_counts = pd.Series(labels).value_counts()
    insufficient_samples_labels = label_counts[label_counts < 2].index.tolist()
    
    train_indices_final = []
    remaining_indices = indices
    if insufficient_samples_labels:
        singleton_mask = np.isin(labels, insufficient_samples_labels)
        train_indices_final.extend(indices[singleton_mask])
        remaining_indices = indices[~singleton_mask]
    
    remaining_labels = labels[remaining_indices]

    temp_size = val_ratio + test_ratio
    train_main_indices, temp_indices, _, _ = train_test_split(
        remaining_indices, remaining_labels, test_size=temp_size, random_state=GLOBAL_SEED, stratify=remaining_labels
    )
    train_indices_final.extend(train_main_indices)
    train_indices = np.array(train_indices_final)

    if len(temp_indices) > 0 and test_ratio > 0:
        relative_test_ratio = test_ratio / temp_size
        if len(temp_indices) < 2:
             val_indices = temp_indices
             test_indices = np.array([], dtype=int)
        else:
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=relative_test_ratio, random_state=GLOBAL_SEED
            )
    else:
        val_indices = temp_indices
        test_indices = np.array([], dtype=int)
        
    return train_indices, val_indices, test_indices


if __name__ == '__main__':
    start_time = time.time()
    
    dataset = CrashDataset(input_file='./data/data_input.npz', label_file='./data/data_labels.npz')
    print(f"\n原始数据加载完成, 耗时: {time.time() - start_time:.2f}s")

    train_indices, val_indices, test_indices = split_data(dataset, train_ratio=0.8, val_ratio=0.19, test_ratio=0.01)
    
    processor = DataProcessor(top_k_waveform=50)
    processor.fit(train_indices, dataset)
    
    processor.print_fit_summary()

    dataset = processor.transform(dataset)
    print("整个数据集已使用训练集统计量完成转换。")

    processor.save('./data/preprocessors.joblib')
    print("处理器已保存至 './data/preprocessors.joblib'")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    torch.save(train_dataset, './data/train_dataset.pt')
    torch.save(val_dataset, './data/val_dataset.pt')
    torch.save(test_dataset, './data/test_dataset.pt')
    print("\n处理后的训练、验证和测试数据集已保存。")

    print(f"\n数据集划分大小:")
    print(f"  - 总计: {len(dataset)}")
    print(f"  - 训练集: {len(train_dataset)}")
    print(f"  - 验证集: {len(val_dataset)}")
    print(f"  - 测试集: {len(test_dataset)}")

    def get_label_distribution(subset):
        if not subset.indices.any(): return "空"
        labels = [subset.dataset.mais[i] for i in subset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

    print("\n各子集的MAIS标签分布:")
    print(f"  - 训练集: {get_label_distribution(train_dataset)}")
    print(f"  - 验证集: {get_label_distribution(val_dataset)}")
    print(f"  - 测试集: {get_label_distribution(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
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
    print(f"batch loading time: {time.time() - batch_start_time:.4f}s")