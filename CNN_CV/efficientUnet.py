import torch
import torch.nn as nn
import timm
from timm.models.efficientnet import tf_efficientnet_b3_ns  # or the specific model you use
from timm.models._efficientnet_blocks import SqueezeExcite


class ConvBlock(nn.Module):
    """Conv → BN → LeakyReLU → SpatialDropout2d"""
    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # after concat, we get out_ch + skip_ch → pass dropout through ConvBlock
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[2:] != x.shape[2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

#%%------------------------------------------------------------------------------------
# EfficientUNet model using EfficientNet as encoder
#------------------------------------------------------------------------------------
class EfficientUNet(nn.Module):
    def __init__(self, in_channels=7, num_classes=7, dropout_p=0.1, pretrained=True, freeze_encoder=False):
        super().__init__()

        # Create encoder with default 3-channel config
        encoder = timm.create_model('efficientnet_b3', pretrained=False, features_only=True)
        self.encoder = encoder

        # Replace first conv layer BEFORE loading pretrained weights
        old_conv = self.encoder.conv_stem
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        self.encoder.conv_stem = new_conv

        if pretrained:
            # Load pretrained weights manually (and ignore mismatched shapes)
            state_dict = timm.create_model('efficientnet_b3', pretrained=True).state_dict()
            # Remove incompatible weights (e.g., conv_stem)
            for k in list(state_dict.keys()):
                if k.startswith("conv_stem") or k.startswith("classifier") or "head" in k:
                    del state_dict[k]
            self.encoder.load_state_dict(state_dict, strict=False)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.enc_channels = [24, 32, 48, 136, 384]
        
        # Decoder blocks
        self.up4 = UpBlock(384, 136, 256, dropout_p=dropout_p)  # 384 from previous layer, 136 from skip
        self.up3 = UpBlock(256, 48, 128, dropout_p=dropout_p)
        self.up2 = UpBlock(128, 32, 64, dropout_p=dropout_p)
        self.up1 = UpBlock(64, 24, 32, dropout_p=dropout_p)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)

        #print([f.shape for f in feats])  # This will print 5 feature maps

        x = self.up4(feats[4], feats[3])
        x = self.up3(x, feats[2])
        x = self.up2(x, feats[1])
        x = self.up1(x, feats[0])
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        return x