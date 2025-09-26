""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # For bilinear upsampling, we need to account for concatenation:
            # in_channels (from previous layer) + out_channels (from skip connection)
            self.conv = DoubleConv(in_channels + out_channels, out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # For transpose convolution: (in_channels // 2) + out_channels = total input channels
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, depth=4, base_channels=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth
        self.base_channels = base_channels
        self.checkpointing = False

        self.inc = DoubleConv(n_channels, base_channels)

        # Encoder
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for _ in range(depth):
            self.downs.append(Down(in_ch, in_ch * 2))
            in_ch *= 2
        
        factor = 2 if bilinear else 1
        self.bottleneck = DoubleConv(in_ch, in_ch * 2 // factor)

        # Decoder - 正确计算通道数
        self.ups = nn.ModuleList()
        # 从bottleneck开始的通道数
        decoder_ch = in_ch * 2 // factor
        # 存储编码器各层的输出通道数，用于skip connection
        encoder_channels = [base_channels]
        temp_ch = base_channels
        for _ in range(depth):
            temp_ch *= 2
            encoder_channels.append(temp_ch)
        
        # 构建解码器层
        for i in range(depth):
            skip_ch = encoder_channels[-(i+2)]  # 对应的skip connection通道数
            out_ch = skip_ch  # 输出通道数等于skip connection的通道数
            self.ups.append(Up(decoder_ch, out_ch, bilinear))
            decoder_ch = out_ch

        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        x = self.inc(x)
        for i in range(self.depth):
            skip_connections.append(x)
            if self.checkpointing:
                x = torch.utils.checkpoint.checkpoint(self.downs[i], x)
            else:
                x = self.downs[i](x)
        
        # Bottleneck
        if self.checkpointing:
            x = torch.utils.checkpoint.checkpoint(self.bottleneck, x)
        else:
            x = self.bottleneck(x)

        # Decoder path
        for i in range(self.depth):
            skip = skip_connections.pop()
            if self.checkpointing:
                # Checkpoint requires a function and its inputs.
                # The lambda function captures the current 'x' and 'skip' for the checkpoint.
                x = torch.utils.checkpoint.checkpoint(lambda inp, sk: self.ups[i](inp, sk), x, skip)
            else:
                x = self.ups[i](x, skip)

        # Output layer
        if self.checkpointing:
            logits = torch.utils.checkpoint.checkpoint(self.outc, x)
        else:
            logits = self.outc(x)
            
        return logits

    def use_checkpointing(self):
        self.checkpointing = True

# # model.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F # 我們需要引入 functional

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         return self.double_conv(x)

# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )
#     def forward(self, x):
#         return self.maxpool_conv(x)

# class Up(nn.Module):
#     """Upscaling then double conv"""
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         # x1 是來自上一個上採樣層的張量
#         # x2 是來自下採樣路徑的、尺寸可能不匹配的張量
#         x1 = self.up(x1)
        
#         # =======================================================
#         # 【核心修改】處理尺寸不匹配問題
#         # =======================================================
#         # F.pad 用於填充，這裡我們計算 x1 和 x2 之間的尺寸差異
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         # 對 x1 進行填充，使其尺寸與 x2 匹配
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # =======================================================
        
#         # 現在尺寸已經匹配，可以安全地進行拼接
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#     def forward(self, x):
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits