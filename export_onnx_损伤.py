import warnings
warnings.filterwarnings('ignore')
import os
import json
import torch
import numpy as np
import argparse
import joblib

# 导入模型定义和数据处理器
from utils import models
from utils.dataset_prepare import DataProcessor

def create_sample_raw_input(raw_input=None):
    """
    为演示和验证目的,创建一个符合物理范围的、未经处理的原始输入样本。
    这模拟了在实际推理场景中,模型接口接收到的原始数据。
    
    输入:
        raw_input (dict, tuple or None): 
            - 如果是字典: 包含参数名和对应值的字典,支持的键包括:
                'impact_velocity', 'impact_angle', 'overlap', 'occupant_type',
                'll1', 'll2', 'btf', 'pp', 'plp', 'lla_status', 'llattf', 
                'dz', 'ptf', 'aft', 'aav_status', 'ttf', 'sp', 'recline_angle',
                'waveform' (形状为 (3, 150) 的碰撞波形)
              缺失或不合规的参数将被随机生成。
            - 如果是元组: 格式为 (raw_params, raw_waveform)
                - raw_params (np.ndarray): 形状为 (18,) 的原始标量特征。
                - raw_waveform (np.ndarray): 形状为 (3, 150) 的原始碰撞波形。
            - 如果为 None: 则随机生成完整样本。

    返回:
        tuple: (raw_params, raw_waveform)
        - raw_params (np.ndarray): 形状为 (18,) 的原始标量特征,float32类型。
                                   顺序与data_package.py中定义的一致。
        - raw_waveform (np.ndarray): 形状为 (3, 150) 的原始碰撞波形,float32类型。
                                     通道顺序为 [X, Y, Z]。
    """
    
    # 定义参数名称与索引的映射
    param_names = [
        'impact_velocity', 'impact_angle', 'overlap', 'occupant_type',
        'll1', 'll2', 'btf', 'pp', 'plp', 'lla_status', 'llattf',
        'dz', 'ptf', 'aft', 'aav_status', 'ttf', 'sp', 'recline_angle'
    ]
    
    # 定义参数验证和生成规则
    def validate_and_generate(key, value, raw_params):
        """验证参数并在不合规时生成随机值"""
        try:
            if key == 'impact_velocity':
                val = float(value)
                return val if 25 <= val <= 65 else np.random.uniform(25, 65)
            
            elif key == 'impact_angle':
                val = float(value)
                return val if -45 <= val <= 45 else np.random.uniform(-45, 45)
            
            elif key == 'overlap':
                val = float(value)
                if (-1 <= val <= -0.25) or (0.25 <= val < 1):
                    return val
                else:
                    return np.random.uniform(0.25, 1.0) if np.random.rand() > 0.5 else np.random.uniform(-1.0, -0.25)
            
            elif key == 'occupant_type':
                val = int(value)
                return val if val in [1, 2, 3] else np.random.choice([1, 2, 3])
            
            elif key == 'll1':
                val = float(value)
                return val if 2.0 <= val <= 7.0 else np.random.uniform(2.0, 7.0)
            
            elif key == 'll2':
                val = float(value)
                return val if 1.5 <= val <= 4.5 else np.random.uniform(1.5, 4.5)
            
            elif key == 'btf':
                val = float(value)
                return val if 10 <= val <= 100 else np.random.uniform(10, 100)
            
            elif key == 'pp':
                val = float(value)
                return val if 40 <= val <= 100 else np.random.uniform(40, 100)
            
            elif key == 'plp':
                val = float(value)
                return val if 20 <= val <= 80 else np.random.uniform(20, 80)
            
            elif key == 'lla_status':
                val = int(value)
                return val if val in [0, 1] else np.random.choice([0, 1])
            
            elif key == 'llattf':
                val = float(value)
                if raw_params[9] == 1:  # lla_status == 1
                    return val if val >= raw_params[6] else raw_params[6] + np.random.uniform(0, 100)
                else:
                    return 0
            
            elif key == 'dz':
                val = int(value)
                return val if val in [1, 2, 3, 4] else np.random.choice([1, 2, 3, 4])
            
            elif key == 'ptf':
                val = float(value)
                # ptf 通常为 btf + 7
                return val if abs(val - (raw_params[6] + 7)) < 20 else raw_params[6] + 7
            
            elif key == 'aft':
                val = float(value)
                return val if 10 <= val <= 100 else np.random.uniform(10, 100)
            
            elif key == 'aav_status':
                val = int(value)
                return val if val in [0, 1] else np.random.choice([0, 1])
            
            elif key == 'ttf':
                val = float(value)
                if raw_params[14] == 1:  # aav_status == 1
                    return val if val >= raw_params[13] else raw_params[13] + np.random.uniform(0, 100)
                else:
                    return 0
            
            elif key == 'sp':
                val = float(value)
                occupant_type = raw_params[3]
                if occupant_type == 1 and 10 <= val <= 110:
                    return val
                elif occupant_type == 2 and -80 <= val <= 80:
                    return val
                elif occupant_type == 3 and -110 <= val <= 40:
                    return val
                else:
                    # 根据occupant_type生成
                    if occupant_type == 1:
                        return np.random.uniform(10, 110)
                    elif occupant_type == 2:
                        return np.random.uniform(-80, 80)
                    else:
                        return np.random.uniform(-110, 40)
            
            elif key == 'recline_angle':
                val = float(value)
                return val if -10 <= val <= 15 else np.random.uniform(-10, 15)
            
        except (ValueError, TypeError):
            # 类型转换失败,返回None触发随机生成
            return None
        
        return None
    
    # 初始化参数数组
    raw_params = np.zeros(18, dtype=np.float32)
    
    # 处理不同类型的输入
    if raw_input is None:
        print("  ⚠ 未提供原始输入,使用随机生成的样本进行演示和验证。")
        input_dict = {}
    elif isinstance(raw_input, dict):
        print("  📥 从字典读取输入参数...")
        input_dict = raw_input
    elif isinstance(raw_input, tuple) and len(raw_input) == 2:
        # 兼容原有的元组输入格式
        print("  📥 从元组读取输入参数...")
        return raw_input[0].astype(np.float32), raw_input[1].astype(np.float32)
    else:
        print("  ⚠ 输入格式不正确,使用随机生成的样本。")
        input_dict = {}
    
    # 按顺序处理每个参数
    for idx, key in enumerate(param_names):
        if key in input_dict:
            validated_value = validate_and_generate(key, input_dict[key], raw_params)
            if validated_value is not None:
                raw_params[idx] = validated_value
                print(f"    ✓ {key}: 使用提供的值 {validated_value:.4f}")
            else:
                # 验证失败,随机生成
                raw_params[idx] = validate_and_generate(key, None, raw_params) or 0
                print(f"    ⚠ {key}: 提供值不合规,已随机生成 {raw_params[idx]:.4f}")
        else:
            # 参数缺失,随机生成
            if key == 'impact_velocity':
                raw_params[idx] = np.random.uniform(25, 65)
            elif key == 'impact_angle':
                raw_params[idx] = np.random.uniform(-45, 45)
            elif key == 'overlap':
                raw_params[idx] = np.random.uniform(0.25, 1.0) if np.random.rand() > 0.5 else np.random.uniform(-1.0, -0.25)
            elif key == 'occupant_type':
                raw_params[idx] = np.random.choice([1, 2, 3])
            elif key == 'll1':
                raw_params[idx] = np.random.uniform(2.0, 7.0)
            elif key == 'll2':
                raw_params[idx] = np.random.uniform(1.5, 4.5)
            elif key == 'btf':
                raw_params[idx] = np.random.uniform(10, 100)
            elif key == 'pp':
                raw_params[idx] = np.random.uniform(40, 100)
            elif key == 'plp':
                raw_params[idx] = np.random.uniform(20, 80)
            elif key == 'lla_status':
                raw_params[idx] = np.random.choice([0, 1])
            elif key == 'llattf':
                raw_params[idx] = raw_params[6] + np.random.uniform(0, 100) if raw_params[9] == 1 else 0
            elif key == 'dz':
                raw_params[idx] = np.random.choice([1, 2, 3, 4])
            elif key == 'ptf':
                raw_params[idx] = raw_params[6] + 7
            elif key == 'aft':
                raw_params[idx] = np.random.uniform(10, 100)
            elif key == 'aav_status':
                raw_params[idx] = np.random.choice([0, 1])
            elif key == 'ttf':
                raw_params[idx] = raw_params[13] + np.random.uniform(0, 100) if raw_params[14] == 1 else 0
            elif key == 'sp':
                if raw_params[3] == 1:
                    raw_params[idx] = np.random.uniform(10, 110)
                elif raw_params[3] == 2:
                    raw_params[idx] = np.random.uniform(-80, 80)
                else:
                    raw_params[idx] = np.random.uniform(-110, 40)
            elif key == 'recline_angle':
                raw_params[idx] = np.random.uniform(-10, 15)
            
            print(f"    ⚠ {key}: 未提供,已随机生成 {raw_params[idx]:.4f}")
    
    # 处理波形数据
    if 'waveform' in input_dict:
        try:
            waveform = np.array(input_dict['waveform'], dtype=np.float32)
            if waveform.shape == (3, 150):
                raw_waveform = waveform
                print("    ✓ waveform: 使用提供的波形数据")
            else:
                print(f"    ⚠ waveform: 形状不正确 {waveform.shape},期望 (3, 150),已随机生成")
                x_wave = -np.abs(np.random.randn(150) * 300 + 100)
                y_wave = np.random.randn(150) * 40
                z_wave = np.random.randn(150) * 20
                raw_waveform = np.stack([x_wave, y_wave, z_wave], axis=0).astype(np.float32)
        except Exception as e:
            print(f"    ⚠ waveform: 解析失败 ({e}),已随机生成")
            x_wave = -np.abs(np.random.randn(150) * 300 + 100)
            y_wave = np.random.randn(150) * 40
            z_wave = np.random.randn(150) * 20
            raw_waveform = np.stack([x_wave, y_wave, z_wave], axis=0).astype(np.float32)
    else:
        print("    ⚠ waveform: 未提供,已随机生成")
        x_wave = -np.abs(np.random.randn(150) * 300 + 100)
        y_wave = np.random.randn(150) * 40
        z_wave = np.random.randn(150) * 20
        raw_waveform = np.stack([x_wave, y_wave, z_wave], axis=0).astype(np.float32)
    
    return raw_params, raw_waveform

def preprocess_input(raw_params, raw_waveform, processor):
    """
    使用加载的DataProcessor对原始输入数据进行预处理，使其符合模型输入要求。

    Args:
        raw_params (np.ndarray): 形状为 (18,) 的原始标量特征。
        raw_waveform (np.ndarray): 形状为 (3, 150) 的原始碰撞波形。
        processor (DataProcessor): 已从'preprocessors.joblib'加载的、拟合好的处理器对象。

    Returns:
        tuple: (x_acc, x_att_continuous, x_att_discrete)
        - x_acc (torch.Tensor): 形状为 (1, 3, 150) 的处理后波形张量。
        - x_att_continuous (torch.Tensor): 形状为 (1, 14) 的处理后连续标量张量。
        - x_att_discrete (torch.Tensor): 形状为 (1, 4) 的处理后离散标量张量, long类型。
    """
    # --- 1. 波形预处理 ---
    # 使用在训练集上学习到的全局因子进行归一化
    waveform_processed = raw_waveform / processor.waveform_norm_factor
    
    # --- 2. 标量特征预处理 ---
    # 将原始输入重塑为 (1, 18) 以匹配scaler和encoder的输入要求
    params_reshaped = raw_params.reshape(1, -1)
    
    # 提取连续和离散特征
    continuous_raw = params_reshaped[:, processor.continuous_indices]
    discrete_raw = params_reshaped[:, processor.discrete_indices]
    
    # 使用加载的scaler转换连续特征
    continuous_processed = np.zeros_like(continuous_raw, dtype=np.float32)
    continuous_processed[:, processor.minmax_indices_in_continuous] = processor.scaler_minmax.transform(continuous_raw[:, processor.minmax_indices_in_continuous])
    continuous_processed[:, processor.maxabs_indices_in_continuous] = processor.scaler_maxabs.transform(continuous_raw[:, processor.maxabs_indices_in_continuous])
    
    # 使用加载的encoder转换离散特征
    discrete_processed = np.zeros_like(discrete_raw, dtype=np.int64)
    for i in range(discrete_raw.shape[1]):
        # 对每个离散特征列进行转换
        discrete_processed[:, i] = processor.encoders_discrete[i].transform(discrete_raw[:, i])

    # --- 3. 转换为PyTorch张量并增加batch维度 ---
    x_acc = torch.tensor(waveform_processed, dtype=torch.float32).unsqueeze(0)
    x_att_continuous = torch.tensor(continuous_processed, dtype=torch.float32)
    x_att_discrete = torch.tensor(discrete_processed, dtype=torch.long)
    
    return x_acc, x_att_continuous, x_att_discrete

def export_model(model, sample_inputs, output_path, model_type="teacher", opset_version=17):
    """
    将PyTorch模型导出为ONNX格式。

    Args:
        model (torch.nn.Module): 待导出的PyTorch模型。
        sample_inputs (tuple): 用于追踪模型图的样本输入张量。
        output_path (str): ONNX文件的保存路径。
        model_type (str): 模型类型, 'teacher' 或 'student'。
        opset_version (int): ONNX算子集版本。
    """
    if model_type == "teacher":
        input_names = ["x_acc", "x_att_continuous", "x_att_discrete"]
        model_inputs = sample_inputs
    else: # student
        input_names = ["x_att_continuous", "x_att_discrete"]
        _, x_att_continuous, x_att_discrete = sample_inputs
        model_inputs = (x_att_continuous, x_att_discrete)

    output_names = ["predictions", "encoder_output", "decoder_output"]
    
    # 定义输入的动态轴，允许batch_size可变
    dynamic_axes = {name: {0: "batch_size"} for name in input_names + output_names}

    torch.onnx.export(
        model,
        model_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset_version,
    )
    print(f"✔ {model_type.capitalize()} 模型已导出至: {output_path}")

    # 简化ONNX模型
    try:
        import onnx
        from onnxsim import simplify
        print("  正在简化ONNX模型...")
        onnx_model = onnx.load(output_path)
        model_simp, check = simplify(onnx_model)
        if check:
            onnx.save(model_simp, output_path)
            print("  ✔ ONNX模型简化完成")
        else:
            print("  ✘ ONNX模型简化验证失败，保留原模型")
    except ImportError:
        print("  ⚠ 未安装onnx-simplifier，跳过简化步骤")

def verify_onnx_model(onnx_path, pytorch_model, sample_inputs, model_type="teacher"):
    """
    使用ONNX Runtime验证导出模型的输出是否与PyTorch模型一致。
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠ 未安装onnxruntime，跳过验证步骤。可运行 'pip install onnxruntime'")
        return
    
    print(f"\n========== 验证 {model_type.capitalize()} ONNX 模型 ==========")
    
    # --- PyTorch 推理 ---
    pytorch_model.eval()
    with torch.no_grad():
        if model_type == "teacher":
            pt_outputs = pytorch_model(*sample_inputs)
        else: # student
            pt_outputs = pytorch_model(sample_inputs[1], sample_inputs[2])
    
    # --- ONNX Runtime 推理 ---
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    if model_type == "teacher":
        onnx_inputs = {
            "x_acc": sample_inputs[0].cpu().numpy(),
            "x_att_continuous": sample_inputs[1].cpu().numpy(),
            "x_att_discrete": sample_inputs[2].cpu().numpy()
        }
    else: # student
        onnx_inputs = {
            "x_att_continuous": sample_inputs[1].cpu().numpy(),
            "x_att_discrete": sample_inputs[2].cpu().numpy()
        }
    
    onnx_outputs = sess.run(None, onnx_inputs)
    
    # --- 打印比较输出 ---
    print("  打印PyTorch和ONNX的输出(只看三个部位损伤值输出, 忽略编码器解码器特征输出):")
    output_names = ["预测值(HIC,Dmax,Nij)"]
    np.set_printoptions(suppress=True, precision=6) # 设置numpy打印选项 避免科学计数法
    for i, name in enumerate(output_names):
        pt_out, onnx_out = pt_outputs[i].cpu().numpy(), onnx_outputs[i]
        print(f"  - {name}:")
        print(f"    PyTorch 输出: {pt_out.flatten()} ")
        print(f"    ONNX 输出:    {onnx_out.flatten()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出教师和学生模型为ONNX格式，并进行验证")
    parser.add_argument("--teacher_run_dir", type=str, required=True, help="教师模型训练目录")
    parser.add_argument("--student_run_dir", type=str, default=None, help="学生模型训练目录（可选）")
    parser.add_argument("--teacher_weight", type=str, default="best_mais_accu.pth", help="教师模型权重文件名")
    parser.add_argument("--student_weight", type=str, default="best_mais_accu.pth", help="学生模型权重文件名")
    parser.add_argument("--output_dir", type=str, default="./onnx_models", help="ONNX模型输出目录")
    parser.add_argument("--processor_path", type=str, default="./data/preprocessors.joblib", help="输入数据的预处理器文件路径")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset版本")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cpu') # ONNX导出和验证建议在CPU上进行以保证一致性

    # 1. 加载预处理器
    print(f"加载预处理器: {args.processor_path}")
    if not os.path.exists(args.processor_path):
        raise FileNotFoundError("错误: 预处理器文件 'preprocessors.joblib' 不存在。\n"
                              "请先运行 'dataset_prepare.py' 脚本来生成此文件。")
    processor = joblib.load(args.processor_path)
    
    # 2. 创建并预处理一个样本输入
    print("\n创建并预处理样本输入...")
    raw_inputs = {
        "impact_velocity": 50,
        "impact_angle": 0,
        "overlap": 0.5,
        "occupant_type": 1,
        "ll1": 5.0,
        "ll2": 3.0,
        "btf": 50,
        "pp": 70,
        "plp": 50,
        "lla_status": 1,
        "llattf": 80,
        "dz": 2,
        "ptf": 57,
        "aft": 50,
        "aav_status": 1,
        "ttf": 70,
        "sp": 50,
        "recline_angle": 5,
        "waveform": -np.random.randn(3, 150) * 200  # 可选自定义波形
    }
    raw_params, raw_waveform = create_sample_raw_input(raw_inputs)
    sample_inputs = preprocess_input(raw_params, raw_waveform, processor)
    
    # 3. 加载并导出教师模型
    print("\n" + "="*50)
    print("处理教师模型")
    print("="*50)
    
    with open(os.path.join(args.teacher_run_dir, "TrainingRecord.json"), "r") as f:
        teacher_params = json.load(f)["hyperparameters"]["model"]
    
    # 从预处理器获取离散特征的类别数
    num_classes_of_discrete = [len(enc.classes_) for enc in processor.encoders_discrete]
    
    teacher_model = models.TeacherModel(**teacher_params, num_classes_of_discrete=num_classes_of_discrete).to(device)
    teacher_model.load_state_dict(torch.load(os.path.join(args.teacher_run_dir, args.teacher_weight), map_location=device))
    teacher_model.eval()
    
    teacher_onnx_path = os.path.join(args.output_dir, "TeacherModel.onnx")
    export_model(teacher_model, sample_inputs, teacher_onnx_path, "teacher", args.opset_version)
    verify_onnx_model(teacher_onnx_path, teacher_model, sample_inputs, "teacher")
    
    # 4. 加载并导出学生模型 (如果提供)
    if args.student_run_dir:
        print("\n" + "="*50)
        print("处理学生模型")
        print("="*50)
        
        with open(os.path.join(args.student_run_dir, "TrainingRecord.json"), "r") as f:
            student_params = json.load(f)["hyperparameters"]["model"]
        
        student_model = models.StudentModel(**student_params, num_classes_of_discrete=num_classes_of_discrete).to(device)
        student_model.load_state_dict(torch.load(os.path.join(args.student_run_dir, args.student_weight), map_location=device))
        student_model.eval()
        
        student_onnx_path = os.path.join(args.output_dir, "StudentModel.onnx")
        export_model(student_model, sample_inputs, student_onnx_path, "student", args.opset_version)
        verify_onnx_model(student_onnx_path, student_model, sample_inputs, "student")

    print("\n" + "="*50)
    print("✔ 所有流程执行完毕！")
    print(f"ONNX 模型已保存在: {args.output_dir}")
    print("="*50)