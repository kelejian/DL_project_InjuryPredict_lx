import numpy as np


def AIS_3_cal_head(HIC, hic_threshold=0.15):
    HIC = np.clip(HIC, 1, 2500)
    coefficients = np.array([
        [1.54, 0.00650],  # P(AIS≥1)
        [3.39, 0.00372]   # P(AIS≥3)
    ])
    threshold = hic_threshold
    c1 = coefficients[:, 0].reshape(-1, 1)
    c2 = coefficients[:, 1].reshape(-1, 1)
    HIC_inv = 200 / HIC
    hic_prob = 1.0 / (1.0 + np.exp(c1 + HIC_inv - c2 * HIC))
    AIS_3 = np.max(np.where(hic_prob.T >= threshold, np.array([1, 3]), 0), axis=1)
    return AIS_3

def AIS_cal_head(HIC, hic_threshold=0.15):
    """
    Calculate AIS level from Head Injury Criterion (HIC).
    Sets AIS to 0 for results less than AIS 1.
    Arguments:
    HIC : array-like
        Head Injury Criterion values.
    hic_threshold : float
        Empirical probability threshold for head injury.
    """
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
    threshold = hic_threshold  # 经验概率阈值

    # 计算 P(AIS≥n) 的概率（向量化计算）
    c1 = coefficients[:, 0].reshape(-1, 1)  # 系数1
    c2 = coefficients[:, 1].reshape(-1, 1)  # 系数2
    HIC_inv = 200 / HIC  # HIC 的倒数部分

    # 计算所有 P(AIS≥n)
    hic_prob = 1.0 / (1.0 + np.exp(c1 + HIC_inv - c2 * HIC))

    # 确定 AIS 等级: 超过阈值的等级中的最高等级
    AIS = np.max(np.where(hic_prob.T >= threshold, np.arange(1, 6), 0), axis=1)

    return AIS

def AIS_cal_chest(Dmax, Dmax_threshold=0.2):
    """
    Calculate AIS level from Chest Displacement (Dmax).
    Sets AIS to 0 for results less than AIS 2.
    Arguments:
    Dmax : array-like
        Chest Displacement values.
    Dmax_threshold : float
        Empirical probability threshold for chest injury.
    """
    # Clip Dmax range to prevent numerical instability
    Dmax = np.clip(Dmax, 0.0, 500.0)

    # Define coefficients [c1, c2] for P(AIS>=n) = 1 / (1 + exp(c1 - c2 * Dmax))
    # Based on document rev_criteria2.pdf, page 73, eq. 4.4
    coefficients = np.array([
        [1.8706, 0.04439],  # P(AIS≥2)
        [3.7124, 0.04750],  # P(AIS≥3)
        [5.0952, 0.04750],  # P(AIS≥4)
        [8.8274, 0.04590]   # P(AIS≥5)
    ])
    threshold = Dmax_threshold  # Empirical probability threshold for chest

    # Calculate P(AIS≥n) probabilities (vectorized)
    c1 = coefficients[:, 0].reshape(-1, 1)  # Intercepts
    c2 = coefficients[:, 1].reshape(-1, 1)  # Slopes
    
    # Calculate all P(AIS≥n) for n=2,3,4,5
    Dmax_prob = 1.0 / (1.0 + np.exp(c1 - c2 * Dmax))

    # Determine AIS level based on threshold. If no threshold is passed (raw_ais=0), AIS is 0.
    # AIS level based on Dmax : (2, 3, 4, 5)
    # raw_ais = np.sum(Dmax_prob.T >= threshold, axis=1)
    # AIS = np.where(raw_ais > 0, raw_ais + 1, 0)
    AIS = np.max(np.where(Dmax_prob.T >= threshold, np.arange(2, 6), 0), axis=1)

    return AIS 

def AIS_cal_neck(Nij, Nij_threshold=0.17):
    """
    Calculate AIS level from Neck Injury Criterion (Nij).
    Sets AIS to 0 for results less than AIS 2.
    Arguments:
    Nij : array-like
        Neck Injury Criterion values.
    Nij_threshold : float
        Empirical probability threshold for neck injury.
    """
    # Clip Nij range to prevent numerical instability
    Nij = np.clip(Nij, 0, 50.0)

    # Define coefficients [c1, c2] for P(AIS>=n) = 1 / (1 + exp(c1 - c2 * Nij))
    # Based on document rev_criteria2.pdf, page 46, eq. 3.2
    coefficients = np.array([
        [2.054, 1.195],  # P(AIS≥2)
        [3.227, 1.969],  # P(AIS≥3)
        [2.693, 1.195],  # P(AIS≥4)
        [3.817, 1.195]   # P(AIS≥5)
    ])
    threshold = Nij_threshold  # Empirical probability threshold for neck

    # Calculate P(AIS≥n) probabilities (vectorized)
    c1 = coefficients[:, 0].reshape(-1, 1)  # Intercepts
    c2 = coefficients[:, 1].reshape(-1, 1)  # Slopes

    # Calculate all P(AIS≥n) for n=2,3,4,5
    nij_prob = 1.0 / (1.0 + np.exp(c1 - c2 * Nij))

    # Determine AIS level based on threshold. If no threshold is passed (raw_ais=0), AIS is 0.
    # AIS level based on Nij : (2, 3, 4, 5)
    # raw_ais = np.sum(nij_prob.T >= threshold, axis=1)
    # AIS = np.where(raw_ais > 0, raw_ais + 1, 0)
    AIS = np.max(np.where(nij_prob.T >= threshold, np.arange(2, 6), 0), axis=1)

    return AIS
