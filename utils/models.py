''' This module includes the occupant injury prediction model. '''

import torch
import torch.nn as nn
from torch_geometric.nn import MLP as PygMLP # 直接用PyG的MLP模块

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        """
        用于 TemporalConvNet 中进行堆叠
        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int): 卷积核大小。
            stride (int): 卷积步幅。
            dropout (float): Dropout 概率。
        """
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) // 2  # 保持输入输出长度一致

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果输入输出通道数不同,使用 1x1 卷积调整维度
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, num_channels, Ksize_init=6, Ksize_mid=3, dropout=0.1, hidden=128):
        """
        教师模型一部分, 负责提取X,Y加速度曲线特征(x_acc), 作为encoder一部分
        Args:
            in_channels (int): 输入通道数。
            num_channels (list): 每个 TemporalBlock 的输出通道数。
            Ksize_init (int): 初始卷积核大小。默认为 6。
            Ksize_mid (int): 中间卷积核大小。默认为 3。
            dropout (float): 默认为 0.1。
            hidden (int): 最终输出的特征维度。默认为 128。
        """
        super(TemporalConvNet, self).__init__()

        kernel_sizes = [Ksize_init] + [Ksize_mid] * (len(num_channels)-1)

        # 确保参数列表长度一致
        assert len(num_channels) == len(kernel_sizes), \
            "参数列表长度必须一致:num_channels, kernel_sizes"
        # 确保kernel_sizes[0]为偶数, 其余为奇数
        assert kernel_sizes[0] % 2 == 0, "kernel_sizes[0]必须为偶数"
        if len(kernel_sizes) > 1:
            assert all([k % 2 == 1 for k in kernel_sizes[1:]]), "kernel_sizes[1:]必须为奇数"

        # 初始卷积层, 并进行一次下采样
        padding_init = (kernel_sizes[0] - 2) // 2  # 保持输入输出长度一致
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, num_channels[0], kernel_size=kernel_sizes[0], stride=2, padding=padding_init),  # 下采样
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU(),
        )

        # 堆叠 TemporalBlock
        layers = []
        in_channels = num_channels[0]
        for i in range(len(num_channels)-1):
            out_channels = num_channels[i+1]
            kernel_size = kernel_sizes[i+1]
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
            in_channels = out_channels  # 更新输入通道数

        self.temporal_blocks = nn.Sequential(*layers)

        # 全局平均池化 + 全连接层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 将时间维度降为 1
        self.fc = nn.Linear(num_channels[-1], hidden)  # 输出特征维度为 hidden

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量,形状为 (B, C, L), C是通道数=2, L是序列长度=150

        Returns:
            torch.Tensor: 输出张量,形状为 (B, hidden)
        """
        # 初始卷积层（下采样）
        x = self.initial_conv(x)  # 输出形状: (B, num_channels[0], L/2)
        # TemporalBlock 堆叠
        x = self.temporal_blocks(x)  # 输出形状: (B, num_channels[-1], L/2)
        # 全局平均池化
        x = self.global_avg_pool(x)  # 输出形状: (B, num_channels[-1], 1)
        # 全连接层
        x = x.squeeze(-1)  # 去掉时间维度,形状: (B, num_channels[-1])
        x = self.fc(x)  # 输出形状: (B, hidden)
        return x

class DiscreteFeatureEmbedding(nn.Module):
    """
    对离散特征进行嵌入处理, 用于教师模型和学生模型的encoder
    输入: x_att_discrete (离散特征), num_classes_of_discrete (每个离散特征的类别数)
    输出: 嵌入后的特征向量 (concat 所有离散特征的嵌入向量)
    """
    def __init__(self, num_classes_of_discrete):
        """
        参数:
            num_classes_of_discrete (list): 每个离散特征的类别数,例如 [7, 2, 2, 3]。
        """
        super(DiscreteFeatureEmbedding, self).__init__()
        
        # 为每个离散特征创建嵌入层
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_classes, num_classes - 1)  # 嵌入维度 = 类别数 - 1
            for num_classes in num_classes_of_discrete
        ])
        
    def forward(self, x_att_discrete):
        """
        对离散特征进行嵌入并拼接。

        参数:
            x_att_discrete (torch.Tensor): 离散特征张量,形状为 (B, num_discrete_features),B 是 batch size。
        
        返回:
            torch.Tensor: 嵌入后的特征向量,形状为 (B, sum(num_classes_of_discrete) - len(num_classes_of_discrete))。
        """
        embedded_features = []
        
        # 对每个离散特征进行嵌入
        for i, embedding_layer in enumerate(self.embedding_layers):
            # 提取第 i 个离散特征 (B, ) -> (B, num_classes - 1)
            feature = x_att_discrete[:, i]
            embedded_feature = embedding_layer(feature)
            embedded_features.append(embedded_feature)
        
        # 拼接所有嵌入后的特征 (B, sum(num_classes_of_discrete) - len(num_classes_of_discrete))
        output = torch.cat(embedded_features, dim=1)
        return output

class TeacherModel(nn.Module):
    def __init__(self, num_classes_of_discrete, 
                 Ksize_init=6, Ksize_mid=3,
                 num_blocks_of_tcn=4, num_layers_of_mlpE=4, num_layers_of_mlpD=4, 
                 mlpE_hidden=128, mlpD_hidden=96, 
                 encoder_output_dim =128, decoder_output_dim=16, 
                 dropout=0.1):
        """
        TeacherModel 的初始化。

        参数:
            num_classes_of_discrete (list): 每个离散特征的类别数。
            Ksize_init (int): TCN 初始卷积核大小。
            Ksize_mid (int): TCN 中间卷积核大小。
            num_blocks_of_tcn (int): TCN 编码器的块数。
            num_layers_of_mlpE (int): MLP 编码器的层数。
            num_layers_of_mlpD (int): MLP 解码器的层数。
            mlpE_hidden (int): MLP 编码器的隐藏层维度。
            mlpD_hidden (int): MLP 解码器的隐藏层维度。
            encoder_output_dim  (int): 编码器的输出特征维度。用于蒸馏。
            decoder_output_dim (int): 解码器的输出特征维度。用于蒸馏。
        """
        super(TeacherModel, self).__init__()

        # 离散特征嵌入层
        self.discrete_embedding = DiscreteFeatureEmbedding(num_classes_of_discrete)

        # TCN 编码器，处理 x_acc
        if num_blocks_of_tcn < 2:
            raise ValueError("num_blocks_of_tcn 必须大于等于 2")
        elif num_blocks_of_tcn >=2 and num_blocks_of_tcn <= 4:
            num_channels = [64, 128] + [256] * (num_blocks_of_tcn - 2)
        elif num_blocks_of_tcn >= 5:
            num_channels = [64, 128] + [256] * (num_blocks_of_tcn - 3) + [512]
        self.tcn = TemporalConvNet(in_channels=2, num_channels=num_channels, Ksize_init=Ksize_init, Ksize_mid=Ksize_mid, hidden=encoder_output_dim  // 2, dropout=dropout)

        # MLP 编码器，处理连续特征和离散特征的嵌入
        if num_layers_of_mlpE < 2:
            raise ValueError("num_layers_of_mlpE 必须大于等于 2")
        mlp_encoder_input_dim = 4 + sum(num_classes_of_discrete) - len(num_classes_of_discrete)  # 连续特征 + 离散特征嵌入 
        self.mlp_encoder = PygMLP(
            in_channels=mlp_encoder_input_dim, 
            hidden_channels=mlpE_hidden,
            out_channels=encoder_output_dim  // 2, # 输出特征维度
            num_layers=num_layers_of_mlpE, # 隐层个数为num_layers-2
            norm="batch_norm",
            act="relu",
            act_first=False, # 先归一化再激活
            plain_last=True, # 最后一层不应用非线性激活、批归一化和 dropout
            dropout=dropout
        )

        self.bn1 = nn.BatchNorm1d(encoder_output_dim  + mlp_encoder_input_dim) # 归一化解码器输入特征
        self.leaky_relu1 = nn.LeakyReLU()

        # MLP 解码器，解码出最终特征
        if num_layers_of_mlpD < 2:
            raise ValueError("num_layers_of_mlpD 必须大于等于 2")
        self.mlp_decoder = PygMLP(
            in_channels=encoder_output_dim  + mlp_encoder_input_dim,  # 复用特征
            hidden_channels=mlpD_hidden,
            out_channels=decoder_output_dim, # 输出特征维度
            num_layers=num_layers_of_mlpD, # 隐层个数为num_layers-2
            norm="batch_norm",
            act="relu",
            act_first=False, # 先归一化再激活
            plain_last=True, # 最后一层不应用非线性激活、批归一化和 dropout
            dropout=dropout
        )

        self.bn2 = nn.BatchNorm1d(decoder_output_dim) # 归一化解码器输出特征
        self.leaky_relu2 = nn.LeakyReLU()
        self.fc = nn.Linear(decoder_output_dim, 1)  # 输出 HIC 预测值
        
        # 初始化所有模型参数
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化所有模型参数"""
        # TCN 初始化
        for m in self.tcn.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 离散特征嵌入层初始化
        for embedding in self.discrete_embedding.embedding_layers:
            nn.init.xavier_uniform_(embedding.weight)

        # MLP编码器和解码器初始化
        for mlp in [self.mlp_encoder, self.mlp_decoder]:
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # BatchNorm层初始化
        for bn in [self.bn1, self.bn2]:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)

        # 最终预测层初始化
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x_acc, x_att_continuous, x_att_discrete):
        """
        参数:
            x_acc (torch.Tensor): 碰撞波形数据，形状为 (B, 2, 150)。
            x_att_continuous (torch.Tensor): 连续特征，形状为 (B, 4)。
            x_att_discrete (torch.Tensor): 离散特征，形状为 (B, 4)。

        返回:
            hic_pred: 预测的 HIC 值，形状为 (B, 1)。
            encoder_output: 编码器的输出，形状为 (B, encoder_output_dim )。
            decoder_output: 解码器的输出，形状为 (B, decoder_output_dim)。
        """
        # 1. 处理离散特征
        x_discrete_embedded = self.discrete_embedding(x_att_discrete) 
        # (B, 4) -> (B, sum(num_classes_of_discrete) - len(num_classes_of_discrete))

        # 2. 处理连续特征和离散特征的嵌入
        x_features = torch.cat([x_att_continuous, x_discrete_embedded], dim=1) # (B, 4 + sum(num_classes_of_discrete) - len(num_classes_of_discrete))
        x_features_encoded = self.mlp_encoder(x_features) # (B, encoder_output_dim  // 2)

        # 3. 编码曲线特征x_acc
        x_acc_encoded = self.tcn(x_acc)  
        # (B, 2, 150) -> (B, encoder_output_dim  // 2)

        # 4. 合并 TCN 和 MLP 的特征, 作为编码器的输出
        encoder_output = torch.cat([x_features_encoded, x_acc_encoded], dim=1) # (B, encoder_output_dim )

        # 5. 解码器输出
        decoder_input = torch.cat([encoder_output, x_features], dim=1) # (B, encoder_output_dim  + 4 + sum(num_classes_of_discrete) - len(num_classes_of_discrete))
        decoder_input = self.bn1(decoder_input)
        decoder_input = self.leaky_relu1(decoder_input)
        decoder_output = self.mlp_decoder(decoder_input)  # (B, decoder_output_dim)

        # 6. 预测 HIC 值
        regression_input = self.bn2(decoder_output)
        regression_input = self.leaky_relu2(regression_input)
        hic_pred = self.fc(regression_input).squeeze(-1) # (B,)

        return hic_pred, encoder_output, decoder_output

class StudentModel(nn.Module):
    def __init__(self, num_classes_of_discrete, 
                 num_layers_of_mlpE=4, num_layers_of_mlpD=4, 
                 mlpE_hidden=128, mlpD_hidden=96, 
                 encoder_output_dim=128, decoder_output_dim=16, 
                 dropout=0.1):
        """
        学生模型的初始化。

        参数:
            num_classes_of_discrete (list): 每个离散特征的类别数。
            num_layers_of_mlpE (int): MLP 编码器的层数。
            num_layers_of_mlpD (int): MLP 解码器的层数。
            mlpE_hidden (int): MLP 编码器的隐藏层维度。
            mlpD_hidden (int): MLP 解码器的隐藏层维度。
            encoder_output_dim (int): 编码器的输出特征维度。需与教师模型一致。
            decoder_output_dim (int): 解码器的输出特征维度。需与教师模型一致。
            dropout (float): Dropout 概率。
        """
        super(StudentModel, self).__init__()

        # 离散特征嵌入层
        self.discrete_embedding = DiscreteFeatureEmbedding(num_classes_of_discrete)

        # MLP 编码器，处理连续特征和离散特征的嵌入
        if num_layers_of_mlpE < 2:
            raise ValueError("num_layers_of_mlpE 必须大于等于 2")
        mlp_encoder_input_dim = 4 + sum(num_classes_of_discrete) - len(num_classes_of_discrete)  # 连续特征 + 离散特征嵌入
        self.mlp_encoder = PygMLP(
            in_channels=mlp_encoder_input_dim, 
            hidden_channels=mlpE_hidden,
            out_channels=encoder_output_dim,  # 输出特征维度与教师模型一致
            num_layers=num_layers_of_mlpE, 
            norm="batch_norm",
            act="relu",
            act_first=False, 
            plain_last=True, 
            dropout=dropout
        )        
        
        self.bn1 = nn.BatchNorm1d(encoder_output_dim  + mlp_encoder_input_dim) # 归一化解码器输入特征
        self.leaky_relu1 = nn.LeakyReLU()

        # MLP 解码器，解码出最终特征
        if num_layers_of_mlpD < 2:
            raise ValueError("num_layers_of_mlpD 必须大于等于 2")
        self.mlp_decoder = PygMLP(
            in_channels=encoder_output_dim + mlp_encoder_input_dim,  # 复用特征
            hidden_channels=mlpD_hidden,
            out_channels=decoder_output_dim,  # 输出特征维度与教师模型一致
            num_layers=num_layers_of_mlpD, 
            norm="batch_norm",
            act="relu",
            act_first=False, 
            plain_last=True, # 最后一层不应用非线性激活、批归一化和 dropout
            dropout=dropout
        )

        self.bn2 = nn.BatchNorm1d(decoder_output_dim) # 归一化解码器输出特征
        self.leaky_relu2 = nn.LeakyReLU()
        self.fc = nn.Linear(decoder_output_dim, 1)  # 输出 HIC 预测值
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        初始化所有模型参数。
        """
        # 离散特征嵌入层初始化
        for embedding in self.discrete_embedding.embedding_layers:
            nn.init.xavier_uniform_(embedding.weight)  # 使用 Xavier 初始化

        # MLP 编码器和解码器初始化
        for mlp in [self.mlp_encoder, self.mlp_decoder]:
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # 使用 He 初始化
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)  # 偏置初始化为 0
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)  # BatchNorm 的 weight 初始化为 1
                    nn.init.constant_(m.bias, 0)  # BatchNorm 的 bias 初始化为 0

        # BatchNorm 层初始化
        for bn in [self.bn1, self.bn2]:
            nn.init.constant_(bn.weight, 1)  # BatchNorm 的 weight 初始化为 1
            nn.init.constant_(bn.bias, 0)  # BatchNorm 的 bias 初始化为 0

        # 最终预测层初始化
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')  # 使用 He 初始化
        nn.init.constant_(self.fc.bias, 0)  # 偏置初始化为 0

    def forward(self, x_att_continuous, x_att_discrete):
        """
        参数:
            x_att_continuous (torch.Tensor): 连续特征，形状为 (B, 4)。
            x_att_discrete (torch.Tensor): 离散特征，形状为 (B, 4)。

        返回:
            hic_pred: 预测的 HIC 值，形状为 (B, 1)。
            encoder_output: 编码器的输出，形状为 (B, encoder_output_dim)。
            decoder_output: 解码器的输出，形状为 (B, decoder_output_dim)。
        """
        # 1. 处理离散特征
        x_discrete_embedded = self.discrete_embedding(x_att_discrete)  # (B, sum(num_classes_of_discrete) - len(num_classes_of_discrete))

        # 2. 处理连续特征和离散特征的嵌入
        x_features = torch.cat([x_att_continuous, x_discrete_embedded], dim=1)  # (B, 4 + sum(num_classes_of_discrete) - len(num_classes_of_discrete))
        encoder_output = self.mlp_encoder(x_features)  # (B, encoder_output_dim)

        # 3. 解码器输出
        decoder_input = torch.cat([encoder_output, x_features], dim=1)  # (B, encoder_output_dim + 4 + sum(num_classes_of_discrete) - len(num_classes_of_discrete))
        decoder_input = self.bn1(decoder_input)
        decoder_input = self.leaky_relu1(decoder_input)
        decoder_output = self.mlp_decoder(decoder_input)  # (B, decoder_output_dim)

        # 4. 预测 HIC 值
        regression_input = self.bn2(decoder_output)
        regression_input = self.leaky_relu2(regression_input)
        hic_pred = self.fc(regression_input).squeeze(-1) # (B,)

        return hic_pred, encoder_output, decoder_output