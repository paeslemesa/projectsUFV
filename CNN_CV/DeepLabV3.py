#%%--------------------------------------------------------------------
# IMPORT
#----------------------------------------------------------------------
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

#%%--------------------------------------------------------------------
# MODELO
#----------------------------------------------------------------------
def deeplabv3_Sentinel3(num_classes=7, in_channels=12):
    """
    Model DeepLabv3 adapted for Sentinel2/MSI sensor
    """
    # Carregar modelo
    model = deeplabv3_resnet50(weights=None)

    # 1. Altera a primeira convolução para entrar 12 canais
    old_conv             = model.backbone.conv1 # Carrega o backbone da primeira convolução

    # Altera parametros da convolução
    model.backbone.conv1 = nn.Conv2d(
        in_channels      = in_channels,              # altera número de canais
        out_channels     = old_conv.out_channels,    # mantem mesmo número de canais de saida
        kernel_size      = old_conv.kernel_size,     # mantem kernel size
        stride           = old_conv.stride,          # mantem stride (passo)
        padding          = old_conv.padding,         # mantem padding
        bias             = old_conv.bias is not None # pega valor de Bias
    )

    # Inicializando os pesos
    nn.init.kaiming_normal_(model.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

    # 2. Muda a cabeça do classificador para o número de canais
    model.classifier[4] = nn.Conv2d(
        in_channels  = model.classifier[4].in_channels,
        out_channels = num_classes,
        kernel_size  = 1
    )

    return model
