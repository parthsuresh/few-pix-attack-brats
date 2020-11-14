import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, 3, padding=1, stride=1),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
        nn.ReLU(inplace=True),
    )


def conv_block_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, 3, padding=1, stride=1),
        nn.BatchNorm3d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
        nn.ReLU(inplace=True),
    )


def init_weights(obj):
    for k, m in obj.named_modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and "reg" in k:
            m.bias.data.fill_(0.5)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_block1 = conv_block(1, 8)
        self.conv_block2 = conv_block(8, 16)
        self.conv_block3 = conv_block(16, 16)
        self.conv_block4 = conv_block(16, 32)
        self.conv_block5 = conv_block(32, 32)
        self.linear = nn.Sequential(
            nn.Linear(32 * 7 * 7 * 4, 1024), nn.ReLU(), nn.Linear(1024, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = x.view(-1, 32 * 7 * 7 * 4)
        x = self.linear(x)
        return x


class BN_Model(nn.Module):
    def __init__(self):
        super(BN_Model, self).__init__()

        self.conv_block1 = conv_block_bn(1, 8)
        self.conv_block2 = conv_block_bn(8, 16)
        self.conv_block3 = conv_block_bn(16, 16)
        self.conv_block4 = conv_block_bn(16, 32)
        self.conv_block5 = conv_block_bn(32, 32)
        self.linear = nn.Sequential(
            nn.Linear(32 * 7 * 7 * 4, 1024), nn.ReLU(), nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = x.view(-1, 32 * 7 * 7 * 4)
        x = self.linear(x)
        return x
