import torch
import torch.nn as nn
import timm


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv2D -> BN -> LeakyReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[2:] != x.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EfficientUNet(nn.Module):
    def __init__(self, in_channels=12, num_classes=7):
        super().__init__()

        # Load EfficientNet encoder from timm
        self.encoder = timm.create_model('efficientnet_b0', pretrained=False, features_only=True)

        # Replace first conv layer to support custom input channels
        old_conv = self.encoder.conv_stem
        self.encoder.conv_stem = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # Encoder stages: [stem, block2, block3, block4, block5]
        self.enc_channels = [16, 24, 40, 112, 320]
        #print([c['num_chs'] for c in self.encoder.feature_info])
        #print(self.encoder.default_cfg['architecture'])

        # Decoder upsampling blocks
        self.up4 = UpBlock(320, 112, 160)
        self.up3 = UpBlock(160, 40, 96)
        self.up2 = UpBlock(96, 24, 64)
        self.up1 = UpBlock(64, 16, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)  # get encoder features

        x = self.up4(feats[4], feats[3])
        x = self.up3(x, feats[2])
        x = self.up2(x, feats[1])
        x = self.up1(x, feats[0])

        x = self.final_conv(x)

        # ✅ Upsample to match label size
        x = nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        return x
