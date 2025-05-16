
#%%-----------------------------------------------------------------------------------------------
# BIBLIOTECAS
#-------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Referência: https://arxiv.org/pdf/2105.15203

#%%-----------------------------------------------------------------------------------------------
# 1. CLASSES DE SUPORTE DO SEGFORMER
#-------------------------------------------------------------------------------------------------


#-----------------------------------------------
# 1.1. Decoder
#-----------------------------------------------
class MLPDecoder(nn.Module):
    def __init__(self, in_channels:int, embed_dim:int, n_classes:int, dropout_rate:float=0.1):
        super().__init__()

        # Define o Multi-Layer Perceptron (MLP) para decodificação
        self.linear_c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias = False), # Convolução 1x1
                nn.BatchNorm2d(embed_dim),                            # Normalização em lote
                nn.ReLU(inplace=True)                                 # Função de ativação ReLU
            ) for c in in_channels
        ])

        # Define a fusão das características dos diferentes canais
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1), # Convolução 1x1
            nn.BatchNorm2d(embed_dim),                                         # Normalização em lote
            nn.ReLU(inplace=True),                                             # Função de ativação ReLU
            nn.Dropout2d(dropout_rate),                                        # Dropout   
            nn.Conv2d(embed_dim, n_classes, kernel_size=1)                     # Convolução 1x1 para saída
        )


    def forward(self, features) -> torch.Tensor:
        """
        Forward pass
        Args:
            features (list): Lista de tensores de características de diferentes camadas do backbone.
        Returns:
            torch.Tensor: Tensor de saída com as características fundidas.
        """
        # Redimensiona as características para o tamanho da imagem de entrada
        # e aplica a convolução 1x1 para cada tensor de características
        resized = []
        target_size = features[0].shape[2:]

        # Redimensiona cada tensor de características para o tamanho da imagem de entrada
        for i, (x, linear) in enumerate(zip(features, self.linear_c)):
            x = linear(x)
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            resized.append(x)

        # Concatena as características redimensionadas ao longo do canal
        x = torch.cat(resized, dim=1)
        # Aplica a fusão das características
        # e retorna o tensor de saída
        return self.fuse(x)


#%%-----------------------------------------------------------------------------------------------
# 2. SEGFORMER LITE
#-------------------------------------------------------------------------------------------------
class SegFormerTransformer(nn.Module):
    """
    SegFormer Transformer model for semantic segmentation.
    Args:
        backbone_name (str): Name of the backbone model from timm.
        n_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        embed_dim (int): Dimension of the embedding space.
    """
    def __init__(self, backbone_name:str='mit_b2',
                 n_channels:int=12,
                 n_classes:int=7,
                 embed_dim:int=256,
                 dropout_rate:float=0.1,
                 pretrained:bool=True):
        """
        Inicializa o modelo SegFormer Transformer.
        Args:
            backbone_name (str): Nome do modelo backbone a ser utilizado.
            n_channels (int): Número de canais de entrada.
            n_classes (int): Número de classes de saída.
            embed_dim (int): Dimensão do espaço de incorporação.
            pretrained (bool): Se True, carrega pesos pré-treinados.
        """
        # Inicializa a classe base
        super().__init__()

        # Carregar o backbone da rede neural pretreinado
        self.backbone = timm.create_model(
                backbone_name,
                features_only=True,
                pretrained=pretrained,
                exportable=True,  # Ensure compatibility with TorchScript
                in_chans=n_channels,  # Número de canais de entrada
                )

        # Redimensionar a camada de entrada do backbone para o número de canais desejado
        #self._modify_input_conv(n_channels)

        # Obter informações sobre as características do backbone
        # e o número de canais de cada camada
        self.feature_info = self.backbone.feature_info
        in_channels = [f["num_chs"] for f in self.feature_info]

        # SegFormer-like MLP decoder
        self.decoder = MLPDecoder(in_channels, embed_dim, n_classes, dropout_rate)


    def forward(self, x):
        features = self.backbone(x)  # List of feature maps
        out = self.decoder(features)
        return out