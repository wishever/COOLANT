import torch
from torch import nn
import torch.nn.functional as F

# Squeeze and Excitation Block Module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels * 2, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool1d(x, 1) # Squeeze
        w = self.fc(x)
        w, b = w.split(w.data.size(1) // 2, dim=1) # Excitation
        w = torch.sigmoid(w)
        
        return x * w + b # Scale and add bias

# Residual Block with SEBlock
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.conv_lower = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        self.conv_upper = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels)
        )

        self.se_block = SEBlock(channels)

    def forward(self, x):
        path = self.conv_lower(x)
        path = self.conv_upper(path)

        path = self.se_block(path)

        path = x + path
        return F.relu(path)

# Network Module
class Network(nn.Module):
    def __init__(self, in_channel, filters, blocks, num_classes):
        super(Network, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channel, filters, 3, padding=1, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks - 1)])

        self.out_conv = nn.Sequential(
            nn.Conv1d(filters, 128, 1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        print("after conv:{}".format(x.shape))
        x = self.res_blocks(x)
        print("after res:{}".format(x.shape))
        x = self.out_conv(x)
        print("after out_conv:{}".format(x.shape))
        x = F.adaptive_avg_pool1d(x, 1)
        print("after avg pool:{}".format(x.shape))

        x = x.view(x.data.size(0), -1)
        print("after view:{}".format(x.shape))
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

net = Network(64, 128, 4, 3)
input_1 = torch.randn(64, 64)
input_2 = torch.randn(64, 64)
input_3 = torch.randn(64, 64)
input_1 = input_1.unsqueeze(-1)
input_2 = input_2.unsqueeze(-1)
input_3 = input_3.unsqueeze(-1)
input = torch.cat([input_1, input_2, input_3], -1)
print(input.shape)

output = net(input)
print("output: {}".format(output.shape))
# print(output)
# attention_0 = output[:,0].unsqueeze(1)
# print(attention_0)
