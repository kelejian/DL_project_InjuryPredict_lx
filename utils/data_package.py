import os
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    from utils.AIS_cal import AIS_3_cal_head, AIS_cal_head, AIS_cal_chest, AIS_cal_neck  # 作为包导入时使用
except ImportError:
    from AIS_cal import AIS_3_cal_head, AIS_cal_head, AIS_cal_chest, AIS_cal_neck   # 直接运行时使用

def package_input_data(pulse_dir, params_path, case_id_list, output_path):
    """
    处理、降采样并将指定案例的输入工况参数和波形数据打包在一起。

    该函数会读取工况参数文件，并根据给定的 case_id 列表，匹配对应的
    原始波形CSV文件。然后将输入参数、输出波形和 case_id 作为一个整体
    保存到一个结构化的 .npz 文件中。

    :param pulse_dir: 存放原始波形CSV文件的目录。
    :param params_path: 包含所有工况参数的 distribution 文件路径 (包含 'case_id' 列)。
    :param case_id_list: 需要处理的案例ID列表。
    :param output_path: 打包后的 .npz 文件保存路径。
    """

    # --- 1. 加载并索引工况参数 ---
    # 读取distribution文件
    if params_path.endswith('.npz'):
        distribution_npz = np.load(params_path, allow_pickle=True)
        params_df = pd.DataFrame({
                key: distribution_npz[key]
                for key in distribution_npz.files
            }).set_index('case_id', drop=False)
    elif params_path.endswith('.csv'):
        params_df = pd.read_csv(params_path)
        params_df.set_index('case_id', inplace=True, drop=False)
    else:
        raise ValueError("Unsupported distribution file format. Use .csv or .npz")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 用于存储最终数据的列表
    processed_case_ids = []
    processed_params = []
    processed_waveforms = []

    print(f"开始处理 {len(case_id_list)} 个案例，将输入和输出打包在一起...")
    for case_id in tqdm(case_id_list, desc="Packaging pulse Data"):
        try:
            # --- 2. 确认参数存在 ---
            if case_id not in params_df.index:
                print(f"警告：在参数文件中未找到案例 {case_id}，已跳过。")
                continue

            # --- 3. 读取并处理波形 ---
            x_path = os.path.join(pulse_dir, f'x{case_id}.csv')
            y_path = os.path.join(pulse_dir, f'y{case_id}.csv')
            z_path = os.path.join(pulse_dir, f'z{case_id}.csv')

            if not all(os.path.exists(p) for p in [x_path, y_path, z_path]):
                print(f"警告：案例 {case_id} 的波形文件不完整，已跳过。")
                continue

            time = pd.read_csv(x_path, sep='\t', header=None, usecols=[0]).values.flatten()
            total_length = len(time)
            dt = np.mean(np.diff(time))
            # ************************************************************************
            if np.isclose(dt, 1e-5, atol=1e-7):
                downsample_indices = np.arange(100, total_length, 100)
            elif np.isclose(dt,  5e-6, atol=5e-8):
                downsample_indices = np.arange(200, total_length, 200)
            else:
                raise ValueError(f"案例 {case_id} 的时间步长 {dt} 不符合预期。")
            # ************************************************************************
            # 读取完整波形数据
            ax_full = pd.read_csv(x_path, sep='\t', header=None, usecols=[1]).values.flatten()
            ay_full = pd.read_csv(y_path, sep='\t', header=None, usecols=[1]).values.flatten()
            az_full = pd.read_csv(z_path, sep='\t', header=None, usecols=[1]).values.flatten()

            ax_sampled = ax_full[downsample_indices]
            ay_sampled = ay_full[downsample_indices]
            az_sampled = az_full[downsample_indices]

            # ************************************************************************
            # 只取前150个点
            ax_sampled = ax_sampled[:150]
            ay_sampled = ay_sampled[:150]
            az_sampled = az_sampled[:150]
            # ************************************************************************
            
            waveforms_np = np.stack([ax_sampled, ay_sampled, az_sampled]).squeeze() # 形状 (3, 150), 通道维度在前，分别是 x, y, z，对应索引 0, 1, 2

            # --- 4. 提取匹配的参数 ---
            params_row = params_df.loc[case_id]
            params_np = np.array([
                params_row['impact_velocity'],
                params_row['impact_angle'],
                params_row['overlap'],
                params_row['occupant_type'],
                params_row['ll1'],
                params_row['ll2'],
                params_row['btf'],
                params_row['pp'],
                params_row['plp'],
                params_row['lla_status'],
                params_row['llattf'],
                params_row['dz'],
                params_row['ptf'],
                params_row['aft'],
                params_row['aav_status'],
                params_row['ttf'],
                params_row['sp'],
                params_row['recline_angle']
            ], dtype=np.float32) # 形状 (18,)

            # --- 5. 添加到结果列表 ---
            processed_case_ids.append(case_id)
            processed_params.append(params_np)
            processed_waveforms.append(waveforms_np)

        except Exception as e:
            print(f"警告：处理案例 {case_id} 时发生错误 '{e}'，已跳过。")
            continue
            
    if not processed_case_ids:
        print("错误：没有成功处理任何数据，未生成输出文件。")
        return

    # --- 6. 将数据列表转换为Numpy数组并保存 ---
    final_case_ids = np.array(processed_case_ids, dtype=int) # 形状 (N,)
    final_params = np.stack(processed_params, axis=0) # 形状 (N, 18)
    final_waveforms = np.stack(processed_waveforms, axis=0) # 形状 (N, 3, 150)

    # 断言检查：输出 case_ids 顺序必须与传入的 case_id_list 一致
    assert np.array_equal(final_case_ids, np.array(case_id_list, dtype=int)), (
        f"case_ids 序列不匹配: 输出{final_case_ids[:5]} vs 输入{case_id_list[:5]}"
    )

    np.savez(
        output_path,
        case_ids=final_case_ids,
        params=final_params,
        waveforms=final_waveforms
    )
    print(f"数据打包完成，已保存至 {output_path}")
    print(f"成功处理并打包的数据数目：{len(final_case_ids)}")
    print(f"打包后文件内容: case_ids shape={final_case_ids.shape}, params shape={final_params.shape}, waveforms shape={final_waveforms.shape}")

if __name__ == '__main__':
    pulse_dir = r'G:\VCS_acc_data\acc_data_before1103_5817'
    params_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1106.csv'
    output_dir = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\DL_project_InjuryPredict\data'
    # 读取distribution文件
    if params_path.endswith('.npz'):
        distribution_npz = np.load(params_path, allow_pickle=True)
        distribution_df = pd.DataFrame({
                key: distribution_npz[key]
                for key in distribution_npz.files
            }).set_index('case_id', drop=False)
    elif params_path.endswith('.csv'):
        distribution_df = pd.read_csv(params_path)
        distribution_df.set_index('case_id', inplace=True, drop=False)
    else:
        raise ValueError("Unsupported distribution file format. Use .csv or .npz")
    
    # 筛选is_pulse_ok和is_injury_ok均为True的行, 并提取对应的case编号和标签
    filtered_df = distribution_df[(distribution_df['is_pulse_ok'] == True) & (distribution_df['is_injury_ok'] == True)]

    case_ids_need = filtered_df['case_id'].astype(int).tolist()
    hic15_labels = filtered_df['HIC15'].astype(float).values
    dmax_labels = filtered_df['Dmax'].astype(float).values
    nij_labels = filtered_df['Nij'].astype(float).values

    # 计算对应的AIS标签
    ais3_head_labels = AIS_3_cal_head(hic15_labels)

    ais_head_labels = AIS_cal_head(hic15_labels)
    ais_chest_labels = AIS_cal_chest(dmax_labels)
    ais_neck_labels = AIS_cal_neck(nij_labels)

    # 计算MAIS
    mais_labels = np.maximum.reduce([ais_head_labels, ais_chest_labels, ais_neck_labels])

    ############################################################################################
    
    # 统计三种损伤值的AIS标签分布
    unique, counts = np.unique(ais_head_labels, return_counts=True)
    label_distribution = dict(zip(unique, counts))
    print(f"AIS头部标签分布: {label_distribution}")

    unique, counts = np.unique(ais_chest_labels, return_counts=True)
    label_distribution = dict(zip(unique, counts))
    print(f"AIS胸部标签分布: {label_distribution}")

    unique, counts = np.unique(ais_neck_labels, return_counts=True)
    label_distribution = dict(zip(unique, counts))
    print(f"AIS颈部标签分布: {label_distribution}")

    unique, counts = np.unique(mais_labels, return_counts=True)
    label_distribution = dict(zip(unique, counts))
    print(f"MAIS标签分布: {label_distribution}")

    ############################################################################################
    # 绘制碰撞速度和HIC/Dmax/Nij的散点图
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # 设定是哪个部位的损伤
    for inj_loc in ['head', 'chest', 'neck']:
        if inj_loc == 'head':
            inj_labels = hic15_labels
            ais_labels = ais_head_labels
        elif inj_loc == 'chest':
            inj_labels = dmax_labels
            ais_labels = ais_chest_labels
        elif inj_loc == 'neck':
            inj_labels = nij_labels
            ais_labels = ais_neck_labels
        else:
            raise ValueError("inj_loc 必须是 'head', 'chest' 或 'neck'")

        # 确保数据顺序一一对应 - 按case_ids_need的顺序提取碰撞速度
        impact_velocities = distribution_df.loc[case_ids_need, 'impact_velocity'].values
        
        # 不同AIS等级使用不同颜色
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'yellow', 'orange', 'red', 'darkred']
        ais_colors = [colors[min(ais, 5)] for ais in ais_labels]
        
        # 创建散点图
        scatter = plt.scatter(impact_velocities, inj_labels, c=ais_colors, alpha=0.6, s=50)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=f'AIS {i}') for i in range(6) if i in ais_labels]
        plt.legend(handles=legend_elements, title='AIS LEVEL', loc='upper left')

        if inj_loc == 'head':
            title = 'impact velocity vs HIC15'
        elif inj_loc == 'chest':
            title = 'impact velocity vs Dmax'
        elif inj_loc == 'neck':
            title = 'impact velocity vs Nij'
        plt.title(title)
        plt.xlabel('impact velocity (km/h)')
        plt.ylabel('Injury Severity Metric')
        plt.grid(True, alpha=0.3)
    plt.show()
    ############################################################################################


    print(f"筛选出的case数量: {len(case_ids_need)}")

    print("\n打包输入数据...")
    package_input_data(
        pulse_dir=pulse_dir,
        params_path=params_path,
        case_id_list=case_ids_need,
        output_path=os.path.join(output_dir, 'data_input.npz')
    )

    print("\n打包标签数据...")
    labels_output_path = os.path.join(output_dir, 'data_labels.npz')
    np.savez(
        labels_output_path,
        case_ids=case_ids_need,
        HIC=hic15_labels,
        Dmax=dmax_labels,
        Nij=nij_labels,
        AIS_head=ais_head_labels,
        AIS_chest=ais_chest_labels,
        AIS_neck=ais_neck_labels,
        MAIS=mais_labels
    )
    print(f"对应的标签已保存至: {labels_output_path}")