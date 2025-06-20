import torch
import torch.nn as nn


class ConvLevel(nn.Module):

    def __init__(self, channels=300, height=3, level=1):
        super().__init__()
        self.channels = channels
        self.level = level

        # 1xH сверточный слой
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=(height, 1),
                              stride=1,
                              padding=0,
                              bias=False)

        with torch.no_grad():
            self.conv.weight.zero_()
            for c in range(channels):
                self.conv.weight[c, c, level, 0] = 1.0

    def forward(self, input):
        B, C, L, H, W, _ = input.shape

        x = input.permute(5, 0, 1, 2, 3, 4).reshape(-1, C, L, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        x = self.conv(x)  # (B*2*L, C, 1, W)
        x = x.reshape(-1, L, C, W).permute(0, 2, 1, 3)
        x = x.reshape(2, B, C, L, W).permute(1, 2, 3, 4, 0)  # (B, C, L, W, 2)
        return x
