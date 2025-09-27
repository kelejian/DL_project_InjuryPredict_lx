import numpy as np

# 经验公式计算AIS
def AIS_3_cal_head(HIC):
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
    AIS_3 = np.max(np.where(hic_prob.T >= threshold, np.array([1, 3]), 0), axis=1)
    return AIS_3

def AIS_cal_head(HIC, prob_output=False):
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

    # 确定 AIS 等级: 超过阈值的等级中的最高等级
    AIS = np.max(np.where(hic_prob.T >= threshold, np.arange(1, 6), 0), axis=1)

    if prob_output:
        # 计算 P(AIS=n) 的概率
        ais_prob = np.zeros((len(HIC), 6))  # 初始化 (样本数, 6)
        ais_prob[:, 0] = 1 - hic_prob[0]  # P(AIS=0)
        ais_prob[:, 1] = hic_prob[0] - hic_prob[1]  # P(AIS=1)
        ais_prob[:, 2] = hic_prob[1] - hic_prob[2]  # P(AIS=2)
        ais_prob[:, 3] = hic_prob[2] - hic_prob[3]  # P(AIS=3)
        ais_prob[:, 4] = hic_prob[3] - hic_prob[4]  # P(AIS=4)
        ais_prob[:, 5] = hic_prob[4]  # P(AIS=5)
        return AIS, ais_prob
    else:
        return AIS

def AIS_cal_chest(C_disp, prob_output=False):
    """
    Calculate AIS level from Chest Displacement (C_disp).
    Sets AIS to 0 for results less than AIS 2.
    """
    # Clip C_disp range to prevent numerical instability
    C_disp = np.clip(C_disp, 0.0, 500.0)

    # Define coefficients [c1, c2] for P(AIS>=n) = 1 / (1 + exp(c1 - c2 * C_disp))
    # Based on document rev_criteria2.pdf, page 73, eq. 4.4
    coefficients = np.array([
        [1.8706, 0.04439],  # P(AIS≥2)
        [3.7124, 0.04750],  # P(AIS≥3)
        [5.0952, 0.04750],  # P(AIS≥4)
        [8.8274, 0.04590]   # P(AIS≥5)
    ])
    threshold = 0.15  # Empirical probability threshold for chest

    # Calculate P(AIS≥n) probabilities (vectorized)
    c1 = coefficients[:, 0].reshape(-1, 1)  # Intercepts
    c2 = coefficients[:, 1].reshape(-1, 1)  # Slopes
    
    # Calculate all P(AIS≥n) for n=2,3,4,5
    cdisp_prob = 1.0 / (1.0 + np.exp(c1 - c2 * C_disp))

    # Determine AIS level based on threshold. If no threshold is passed (raw_ais=0), AIS is 0.
    # AIS level based on cdisp : (2, 3, 4, 5)
    # raw_ais = np.sum(cdisp_prob.T >= threshold, axis=1)
    # AIS = np.where(raw_ais > 0, raw_ais + 1, 0)
    AIS = np.max(np.where(cdisp_prob.T >= threshold, np.arange(2, 6), 0), axis=1)

    if prob_output:
        # Calculate P(AIS=n) probabilities
        ais_prob = np.zeros((len(C_disp), 6))  # Initialize for AIS levels 0 through 5
        ais_prob[:, 0] = 1 - cdisp_prob[0]        # P(AIS<2)
        ais_prob[:, 1] = 0                        # No data available for P(AIS=1)
        ais_prob[:, 2] = cdisp_prob[0] - cdisp_prob[1] # P(AIS=2)
        ais_prob[:, 3] = cdisp_prob[1] - cdisp_prob[2] # P(AIS=3)
        ais_prob[:, 4] = cdisp_prob[2] - cdisp_prob[3] # P(AIS=4)
        ais_prob[:, 5] = cdisp_prob[3]            # P(AIS=5)
        return AIS, ais_prob
    else:
        return AIS 

def AIS_cal_neck(Nij, prob_output=False):
    """
    Calculate AIS level from Neck Injury Criterion (Nij).
    Sets AIS to 0 for results less than AIS 2.
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
    threshold = 0.17  # Empirical probability threshold for neck

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

    if prob_output:
        # Calculate P(AIS=n) probabilities
        ais_prob = np.zeros((len(Nij), 6))  # Initialize for AIS levels 0 through 5
        ais_prob[:, 0] = 1 - nij_prob[0]      # P(AIS<2)
        ais_prob[:, 1] = 0                    # No data available for P(AIS=1)
        ais_prob[:, 2] = nij_prob[0] - nij_prob[1] # P(AIS=2)
        ais_prob[:, 3] = nij_prob[1] - nij_prob[2] # P(AIS=3)
        ais_prob[:, 4] = nij_prob[2] - nij_prob[3] # P(AIS=4)
        ais_prob[:, 5] = nij_prob[3]          # P(AIS=5)
        return AIS, ais_prob
    else:
        return AIS
