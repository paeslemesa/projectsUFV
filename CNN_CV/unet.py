
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%-----------------------------------------------------------------------------------------------
# 1. CLASSES DE SUPORTE DA UNET
#-------------------------------------------------------------------------------------------------

#-----------------------------------------------
# 1.1. Classe de convolução dupla
#-----------------------------------------------

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Primeira convolução
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # Convolução 2D 3x3
            nn.BatchNorm2d(out_channels), # Normalização em lote
            nn.ReLU(inplace=True), # Função de ativação ReLU
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(), # Dropout opcional

            # Segunda convolução
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # Convolução 2D 3x3
            nn.BatchNorm2d(out_channels), # Normalização em lote
            nn.ReLU(inplace=True) # Função de ativação ReLU
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity() # Dropout opcional
            )
    
    def forward(self, x):
        """Forward pass
        x: tensor de entrada
        Aplica a sequência de convoluções, normalizações e ativações
        e retorna o resultado
        """
        return self.double_conv(x)
#-----------------------------------------------
# 1.2. Classe de downsampling
#-----------------------------------------------

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # MaxPooling 2D com tamanho de kernel 2
            DoubleConv(in_channels, out_channels, dropout) # Convolução dupla
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

#-----------------------------------------------
# 1.2. Classe de upsampling
#-----------------------------------------------
class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        """
        in_channels : número de canais de entrada
        out_channels: número de canais de saída
        bilinear    : se True, usa upsampling bilinear
        dropout     : taxa de dropout
        """
        super().__init__()
        
        if bilinear: # Se bilinear for verdadeiro, usa upsampling bilinear
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
        else: # Caso contrário, usa convolução transposta
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
    
    def forward(self, x1, x2):
        x1 = self.up(x1) # Aumenta a resolução do tensor x1
        diffY = x2.size()[2] - x1.size()[2] # Diferença de altura
        diffX = x2.size()[3] - x1.size()[3] # Diferença de largura
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # Preenche o tensor x1 para que tenha o mesmo tamanho que x2
        x = torch.cat([x2, x1], dim=1) # Concatena as tensores ao longo do canal
        return self.conv(x)


#%%-----------------------------------------------------------------------------------------------
# 2. ARQUITECTURA UNET
#-------------------------------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, n_channels=12, n_classes=7, bilinear=True, dropout=0.1):
        super(UNet, self).__init__()
        """
        n_channels : número de canais de entrada
        n_classes  : número de classes de saída
        bilinear   : se True, usa upsampling bilinear
        dropout    : taxa de dropout
        """

        # Inicializa os parâmetros
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear
        self.dropout    = dropout
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64,  128, self.dropout)
        self.down2 = Down(128, 256, self.dropout)
        self.down3 = Down(256, 512, self.dropout)
        self.down4 = Down(512, 512, self.dropout)

        # Decoder
        self.up1 = Up(1024, 256, bilinear, self.dropout)
        self.up2 = Up(512,  128, bilinear, self.dropout)
        self.up3 = Up(256,   64, bilinear, self.dropout)
        self.up4 = Up(128,   64, bilinear, self.dropout)
        
        # Convolução final
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # Inicializar os pesos para a rede
        self._initialize_weights()
    
    def forward(self, x):
        """Forward pass
        x: tensor de entrada
        Aplica a sequência de convoluções, normalizações e ativações
        e retorna o resultado
        """
        # Convolução inicial
        x1 = self.inc(x)
        # Downsampling
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Upsampling
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        # Convolução final
        logits = self.outc(x)
        return logits


    def _initialize_weights(self):
        """Inicializa os pesos da rede"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)