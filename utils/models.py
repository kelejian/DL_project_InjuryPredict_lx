''' This module includes the occupant injury prediction model. '''

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2,
                                 self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class teacher_model(nn.Module):
    def __init__(self, embed_size, num_channels, kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(teacher_model, self).__init__()
        self.encoder_x = nn.Embedding(200, embed_size)
        self.encoder_y = nn.Embedding(200, embed_size)
        self.tcn = TemporalConvNet(embed_size * 2, num_channels, kernel_size, dropout=dropout)
        self.encoder_z1 = nn.Embedding(2, embed_size)
        self.encoder_z2 = nn.Embedding(2, embed_size)
        self.encoder_z3 = nn.Embedding(3, embed_size)
        self.decoder = MLP(embed_size, 16, 64)
        self.linear_f = nn.Linear(16, 1)
        self.init_weights()

    def init_weights(self):
        self.encoder_x.weight.data.normal_(0, 0.01)
        self.encoder_z1.weight.data.normal_(0, 0.01)
        self.encoder_z2.weight.data.normal_(0, 0.01)
        self.encoder_z3.weight.data.normal_(0, 0.01)

    def forward(self, x_acc, x_att):
        x_acc_emb_1 = self.encoder_x(x_acc[:, 0]) # -> (batch_size, T, embed_size)
        x_acc_emb_2 = self.encoder_y(x_acc[:, 1]) # (batch_size, T, embed_size)
        x_acc_emb = torch.cat([x_acc_emb_1, x_acc_emb_2], 2) # (batch_size, T, 2 * embed_size)

        output = self.tcn(x_acc_emb.transpose(1, 2)).transpose(1, 2).contiguous() # (batch_size, T, num_channels[-1])

        x = torch.mean(output, 1, keepdim=False) # (batch_size, num_channels[-1])
        emb_z1 = self.encoder_z1(x_att[:, 0])
        emb_z2 = self.encoder_z2(x_att[:, 1])
        emb_z3 = self.encoder_z3(x_att[:, 2])
        emb_extended = x + emb_z1 + emb_z2 + emb_z3

        outputs = self.decoder(emb_extended)  # (batch_size, 16)

        return self.linear_f(outputs).squeeze(1), outputs, x


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden=64, bias=True, activation="relu", norm='layer'):
        super(MLP, self).__init__()

        # define the activation function
        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "relu6":
            act_layer = nn.ReLU6
        elif activation == "leaky":
            act_layer = nn.LeakyReLU
        elif activation == "prelu":
            act_layer = nn.PReLU
        else:
            raise NotImplementedError

        # define the normalization function
        if norm == "layer":
            norm_layer = nn.LayerNorm
        elif norm == "batch":
            norm_layer = nn.BatchNorm1d
        else:
            raise NotImplementedError

        # insert the layers
        self.linear1 = nn.Linear(in_channel, hidden, bias=bias)
        self.linear1.apply(self._init_weights)
        self.linear2 = nn.Linear(hidden, out_channel, bias=bias)
        self.linear2.apply(self._init_weights)

        self.norm1 = norm_layer(hidden)
        self.norm2 = norm_layer(out_channel)

        self.act1 = act_layer(inplace=True)
        self.act2 = act_layer(inplace=True)

        self.shortcut = None
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=bias),
                norm_layer(out_channel)
            )

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
        return self.act2(out)


class student_model(nn.Module):
    def __init__(self, embed_size):
        super(student_model, self).__init__()
        self.encoder_x1 = nn.Embedding(30, embed_size)
        self.encoder_x2 = nn.Embedding(20, embed_size)
        self.encoder_x3 = nn.Embedding(7, embed_size)
        self.encoder_x4 = nn.Embedding(13, embed_size)
        self.encoder_x5 = nn.Embedding(6, embed_size)
        self.encoder = MLP(embed_size, embed_size, embed_size)
        self.encoder_z1 = nn.Embedding(2, embed_size)
        self.encoder_z2 = nn.Embedding(2, embed_size)
        self.encoder_z3 = nn.Embedding(3, embed_size)
        self.decoder = MLP(embed_size, 16, 64)
        self.linear_f = nn.Linear(16, 1)
        self.init_weights()

    def init_weights(self):
        self.encoder_x1.weight.data.normal_(0, 0.01)
        self.encoder_x2.weight.data.normal_(0, 0.01)
        self.encoder_x3.weight.data.normal_(0, 0.01)
        self.encoder_x4.weight.data.normal_(0, 0.01)
        self.encoder_x5.weight.data.normal_(0, 0.01)
        self.encoder_z1.weight.data.normal_(0, 0.01)
        self.encoder_z2.weight.data.normal_(0, 0.01)
        self.encoder_z3.weight.data.normal_(0, 0.01)

    def forward(self, x_att):
        emb_x1 = self.encoder_x1(x_att[:, 0])
        emb_x2 = self.encoder_x2(x_att[:, 1])
        emb_x3 = self.encoder_x3(x_att[:, 2])
        emb_x4 = self.encoder_x4(x_att[:, 3])
        emb_x5 = self.encoder_x5(x_att[:, 4])
        x = emb_x1 + emb_x2 + emb_x3 + emb_x4 + emb_x5 # (batch_size, embed_size)
        x = self.encoder(x) # (batch_size, embed_size)

        emb_z1 = self.encoder_z1(x_att[:, 0 + 5])
        emb_z2 = self.encoder_z2(x_att[:, 1 + 5])
        emb_z3 = self.encoder_z3(x_att[:, 2 + 5])
        emb_extended = x + emb_z1 + emb_z2 + emb_z3 # (batch_size, embed_size)

        outputs = self.decoder(emb_extended) # (batch_size, 16)

        return self.linear_f(outputs).squeeze(1), outputs, x
