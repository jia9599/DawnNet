""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet_Uncertainty_V2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_Uncertainty_V2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down6 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.up5 = Up(64, 32, bilinear)
        self.up6 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

        # 蒙特卡洛Dropout层（添加到关键位置）
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=0.2),  # 浅层低丢弃率
            nn.Dropout(p=0.3),
            nn.Dropout(p=0.4),  # 深层高丢弃率
            nn.Dropout(p=0.4),
            nn.Dropout(p=0.3),  # 解码器中等丢弃率
            nn.Dropout(p=0.3)
        ])

    def forward(self, x, apply_drop=False):
        # Encoder
        if apply_drop:
            x1 = self.inc(x)
            x2 = self.down1(x1)

            x3 = self.down2(x2)


            x4 = self.down3(x3)



            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.down6(x6)

            x7 = F.dropout(x7, 0.4, training=True)

            # Decoder
            x = self.up1(x7, x6)
            x = F.dropout(x, 0.3, training=True)
            x = self.up2(x, x5)
            x = F.dropout(x, 0.3, training=True)
            x = self.up3(x, x4)
            x = F.dropout(x, 0.3, training=True)
            x = self.up4(x, x3)
            x = F.dropout(x, 0.3, training=True)
            x = self.up5(x, x2)
            x = F.dropout(x, 0.3, training=True)
            x = self.up6(x, x1)

            logits = self.outc(x)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.down6(x6)
            x = self.up1(x7, x6)
            x = self.up2(x, x5)
            x = self.up3(x, x4)
            x = self.up4(x, x3)
            x = self.up5(x, x2)
            x = self.up6(x, x1)
            logits = self.outc(x)

        return logits

# class UNet_Uncertainty(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet_Uncertainty, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = (DoubleConv(n_channels, 64))
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
#
#         # self.dropout = nn.Dropout(p=0.3)  # 定义 Dropout 层
#     def forward(self, x, apply_drop=False, dropout_ratio=0.3):
#         if apply_drop:
#             x1 = self.inc(x)
#             x2 = self.down1(x1)
#             x2 = F.dropout(x2,dropout_ratio)
#             x3 = self.down2(x2)
#             x4 = self.down3(x3)
#             x4 = F.dropout(x4, dropout_ratio)
#             x5 = self.down4(x4)
#             x = self.up1(x5, x4)
#             x = F.dropout(x, dropout_ratio)
#             x = self.up2(x, x3)
#             x = self.up3(x, x2)
#             x = F.dropout(x,dropout_ratio)
#             x = self.up4(x, x1)
#             logits = self.outc(x)
#         else:
#             x1 = self.inc(x)
#             x2 = self.down1(x1)
#             x3 = self.down2(x2)
#             x4 = self.down3(x3)
#             x5 = self.down4(x4)
#             x = self.up1(x5, x4)
#             x = self.up2(x, x3)
#             x = self.up3(x, x2)
#             x = self.up4(x, x1)
#             logits = self.outc(x)
#
#         return logits
#
#     def use_checkpointing(self):
#         self.inc = torch.utils.checkpoint(self.inc)
#         self.down1 = torch.utils.checkpoint(self.down1)
#         self.down2 = torch.utils.checkpoint(self.down2)
#         self.down3 = torch.utils.checkpoint(self.down3)
#         self.down4 = torch.utils.checkpoint(self.down4)
#         self.up1 = torch.utils.checkpoint(self.up1)
#         self.up2 = torch.utils.checkpoint(self.up2)
#         self.up3 = torch.utils.checkpoint(self.up3)
#         self.up4 = torch.utils.checkpoint(self.up4)
#         self.outc = torch.utils.checkpoint(self.outc)