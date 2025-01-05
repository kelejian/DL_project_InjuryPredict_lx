import torch
import torch.nn as nn
import torch.onnx
from torchinfo import summary
from torchviz import make_dot
import onnx
import onnxruntime

#@ profile
def test_model(
    model, 
    inputs, 
    labels, 
    criterion=None, 
    optimizer=None, 
    onnx_file_path="model_test.onnx"
):
    """
    通用化模型测试函数：
    1. 接受任意模型实例化对象 `model`。
    2. 自定义输入 `inputs` 和标签 `labels`。
    3. 支持前向传播、反向传播、损失计算。
    4. 导出 ONNX 模型并验证。
    5. 输出模型详细信息。
    
    参数：
    - model: PyTorch 模型实例化对象: torch.nn.Module
    - inputs: 模型的输入张量: tensor 或 tulple(tensor1, tensor2, ...) 或 list(tensor1, tensor2, ...)
    - labels: 模型的真实标签张量（用于损失计算）: tensor
    - criterion: 损失函数实例化对象，默认为 nn.MSELoss
    - optimizer: 优化器实例化对象，默认为 Adam
    - onnx_file_path: 导出的 ONNX 文件路径
    """
    # 默认损失函数和优化器
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 将模型设为训练模式
    model.train()
    print("\n~~~~~~~~~~~~~~~~~~~ 🚀🚀 开始测试神经网络模型是否可以正常训练 🚀🚀 ~~~~~~~~~~~~~~~~~~~~")
    # 打印模型结构信息
    print("\n============== 模型结构信息 ==============")
    _input_data = tuple(inputs) if isinstance(inputs, (tuple, list)) else inputs
    summary(
        model,
        input_data=_input_data,
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
        device="cuda" if next(model.parameters()).is_cuda else "cpu"
    )
    
    # 前向传播与loss计算
    print("\n============== 前向传播 ==============")
    if isinstance(inputs, (tuple, list)):
        outputs = model(*inputs)
        # 一行打印模型各个输入input的形状
        print(f"✔ 模型各个输入的形状：{[input.shape for input in inputs]}")

    else:
        outputs = model(inputs)
        print(f"✔ 输入形状：{inputs.shape}")

    if isinstance(outputs, (tuple, list)):
        # 一行打印模型各个输出output的形状
        print(f"✔ 模型各个输出的形状：{[output.shape for output in outputs]}")
        for i, output in enumerate(outputs):
            if labels.shape == output.shape:
                loss = criterion(output, labels)
                print(f"✔ 第{i+1}个模型输出对应了一个loss值: {loss.item()}")

    else:
        print(f"✔ 模型输出形状：{outputs.shape}")
        if labels.shape == outputs.shape:
            loss = criterion(outputs, labels)
            print(f"✔ 损失值：{loss.item()}")
        else: 
            print("✘ 模型输出形状与标签形状不匹配，无法计算损失值")


    # 反向传播
    print("\n============== 反向传播 ==============")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("✔ 反向传播正常~")

    # 可视化计算图
    print("\n============== 计算图可视化 ==============")
    graph = make_dot(loss, params=dict(model.named_parameters()))
    graph.render("model_computation_graph", format="png")
    print("✔ 计算图已保存为 'model_computation_graph.png'")

    # 导出 ONNX 模型
    print("\n============== 导出 ONNX 模型 ==============")
    torch.onnx.export(
        model,
        _input_data,
        onnx_file_path,
        input_names=[f"input_{i}" for i in range(len(inputs))],
        output_names=["output"],
        dynamic_axes={f"input_{i}": {0: "batch_size"} for i in range(len(inputs))},
        opset_version=11,
    )
    print(f"✔ ONNX 模型已保存至 {onnx_file_path}")
    print("在 https://netron.app/ 上查看 ONNX 模型结构")

    # 验证 ONNX 模型
    print("\n============== 验证 ONNX 模型 ==============")
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print("✔ ONNX 模型验证成功！")

    # # 使用 ONNX Runtime 推理
    # print("\n============== ONNX Runtime 推理 ==============")
    # ort_session = onnxruntime.InferenceSession(onnx_file_path)
    # ort_inputs = {onnx_model.graph.input[i].name: inputs[i].cpu().numpy() if isinstance(inputs, (tuple, list)) else inputs.cpu().numpy()
    #               for i in range(len(onnx_model.graph.input))}
    # ort_outs = ort_session.run(None, ort_inputs)
    # print(f"ONNX 推理输出：{ort_outs}")


# 示例用法
if __name__ == "__main__":
    from utils.models import TeacherModel  # 导入最新版的教师模型

    # 模型实例化
    num_classes_of_discrete = [7, 2, 2, 3]  # 离散特征的类别数
    model = TeacherModel(num_classes_of_discrete).cuda()  # 使用最新版的教师模型

    # 示例输入数据（模拟数据集第1个batch）
    batch_size = 128
    time_steps = 150

    # x_acc 输入，形状 (batch_size, 2, time_steps)
    x_acc = torch.randn(batch_size, 2, time_steps).cuda()  # 随机生成加速度曲线数据

    # x_att_continuous 输入，形状 (batch_size, 4)
    x_att_continuous = torch.randn(batch_size, 4).cuda()  # 随机生成连续特征

    # x_att_discrete 输入，形状 (batch_size, 4)
    x_att_discrete = torch.cat([
        torch.randint(0, 7, (batch_size, 1)),  # collision overlap rate (0~6)
        torch.randint(0, 2, (batch_size, 1)),  # belt usage (0/1)
        torch.randint(0, 2, (batch_size, 1)),  # airbag usage (0/1)
        torch.randint(0, 3, (batch_size, 1))   # occupant size (0/1/2)
    ], dim=1).cuda()  # 随机生成离散特征

    # 模拟真实标签
    y_HIC = torch.randn(batch_size, 1).cuda()  # 随机生成 HIC 标签

    # 测试模型
    test_model(model, inputs=(x_acc, x_att_continuous, x_att_discrete), labels=y_HIC)