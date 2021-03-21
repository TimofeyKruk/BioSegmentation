""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n0 = 32
        n1 = n0 * 2
        n2 = n0 * 4
        n3 = n0 * 8
        n4 = n0 * 16

        self.inc0_0 = DoubleConv(n_channels, n0)

        # Encoder
        self.down1_0 = Down(n0, n1)
        self.down2_0 = Down(n1, n2)
        self.down3_0 = Down(n2, n3)
        # factor = 2 if bilinear else 1
        self.down4_0 = Down(n3, n4)

        # Skip
        # parallel to encoder
        self.skip0_1 = Up(n1 + n0, n0, bilinear)
        self.skip1_1 = Up(n2 + n1, n1, bilinear)
        self.skip2_1 = Up(n3 + n2, n2, bilinear)
        # next
        self.skip0_2 = Up(n1 + 2 * n0, n0, bilinear)
        self.skip1_2 = Up(n2 + 2 * n1, n1, bilinear)
        # next
        self.skip0_3 = Up(n1 + 3 * n0, n0, bilinear)

        # Decoder
        self.up3_1 = Up(n4 + 1 * n3, n3, bilinear)
        self.up2_2 = Up(n3 + 2 * n2, n2, bilinear)
        self.up1_3 = Up(n2 + 3 * n1, n1, bilinear)
        self.up0_4 = Up(n1 + 4 * n0, n0, bilinear)
        # self.up1 = Up(1024, 512 , bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(n0, n_classes)

    def forward(self, x):
        # down
        x0_0 = self.inc0_0(x)
        x1_0 = self.down1_0(x0_0)
        x2_0 = self.down2_0(x1_0)
        x3_0 = self.down3_0(x2_0)
        x4_0 = self.down4_0(x3_0)

        # Skip+decoder
        # parallet to encoder
        x0_1 = self.skip0_1(x1_0, x0_0)
        x1_1 = self.skip1_1(x2_0, x1_0)
        x2_1 = self.skip2_1(x3_0, x2_0)
        x3_1 = self.up3_1(x4_0, x3_0)
        # next
        x0_2 = self.skip0_2(x1_1, torch.cat([x0_0, x0_1], dim=1))
        x1_2 = self.skip1_2(x2_1, torch.cat([x1_0, x1_1], dim=1))
        x2_2 = self.up2_2(x3_1, torch.cat([x2_0, x2_1], dim=1))
        # next
        x0_3 = self.skip0_3(x1_2, torch.cat([x0_0, x0_1, x0_2], dim=1))
        x1_3 = self.up1_3(x2_2, torch.cat([x1_0, x1_1, x1_2], dim=1))
        # next
        x0_4 = self.up0_4(x1_3, torch.cat([x0_0, x0_1, x0_2, x0_3], dim=1))

        logits = self.outc(x0_4)
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        return logits
