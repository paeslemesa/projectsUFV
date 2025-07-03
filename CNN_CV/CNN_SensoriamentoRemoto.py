#%%------------------------------------------------------------------------------------
# BIBLIOTECAS
#------------------------------------------------------------------------------------
import os

# PyTorch
import torch
from torch.utils.data import random_split, DataLoader, Subset
#import torch.multiprocessing as mp

# Visualização de modelo
#from torchinfo import summary
#from torchview import draw_graph

# Visualização e análise
import matplotlib.pyplot as plt
import numpy as np

# Dataset
from CerraDataDataset import CerraDataset

# Modelo
#from DeepLabV3 import deeplabv3_Sentinel3
from efficientUnet import EfficientUNet
from unet import UNet

# Funções auxiliares
import funcao_treino
import losses


#%%------------------------------------------------------------------------------------
# ENTRADA DE DADOS
#------------------------------------------------------------------------------------

# Caminho dos dados
caminho = "/home/sabrina/Documents/Datasets/cerradata_4mm/"

# Hiperparâmetros
batch_size       = 128
msi_bands        = [3,2,1,4] # Bandas MSI: RGB + NIR
num_workers      = 10  # or 4, or 8 based on your system
n_classes        = 7
veg_indexes      = True  # Calcular EVI2, NDVI e SAVI
sar_bands       =  False  # Bandas SAR: VV e VH
height, width    = 128, 128
epocas           = 100
taxa_aprendizagem= 1e-2
taxa_decaimento  = 1e-3 
n_samples        = None # None para usar todo o dataset
transforms       = True
pretreino        = False  # Usar pesos pré-treinados do EfficientNet
modelo           = "UNet"  # Modelo a ser usado: "EfficientUNet" ou "UNet"
n_canais         = len(msi_bands) + (2 if sar_bands else 0) + (3 if veg_indexes else 0)



#%%------------------------------------------------------------------------------------
# VERIFICAÇÃO DE HIPERPARÂMETROS
if pretreino:
    taxa_aprendizagem = taxa_aprendizagem * 1e-2  # Reduzir taxa de aprendizado se usar pesos pré-treinados
    print("Usando pesos pré-treinados do EfficientNet. Taxa de aprendizado ajustada para:", taxa_aprendizagem)

# Verificar se há GPU disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Criar um dicionário de hiperparâmetros
hiperparametros = {
    "batch_size": batch_size,
    "num_workers": num_workers,
    "msi_bands": ",".join(str(element) for element in msi_bands),
    "sar_bands": sar_bands,  # Bandas SAR (2 bandas)
    "n_classes": n_classes,
    "n_canais": n_canais,
    "veg_indexes": veg_indexes,
    "device": str(device),
    "height": height,
    "width": width,
    "epocas": epocas,
    "taxa_aprendizagem": taxa_aprendizagem,
    "taxa_decaimento": taxa_decaimento,
    "n_samples": n_samples,
    "transforms": transforms,
    "pretreino": pretreino,
    "modelo": modelo
}


#%%------------------------------------------------------------------------------------
# DATASETS E DATALOADERS
#------------------------------------------------------------------------------------

# Carregar dataset
dataset = CerraDataset(
    cam_dir=caminho, 
    dispositivo=device, 
    normalizacao='1a1', 
    transformar=transforms,
    bands=msi_bands,  # RGB + NIR
    sar_bands=sar_bands,  # Bandas SAR: VV e VH
    veg_indexes=veg_indexes,  # Calcular EVI2, NDVI e SAVI
)

if dataset is None or len(dataset) == 0:
    raise ValueError("Dataset não carregado corretamente ou está vazio.")

if n_samples is not None:
    n_samples = min(n_samples, len(dataset))
    subset_indices = list(range(n_samples))
    dataset = Subset(dataset, subset_indices)


# Dividir em treino e validação
razao_treino = 0.8
tam_treino = int(razao_treino * len(dataset))
tam_val   = len(dataset) - tam_treino

train_dataset, val_dataset = random_split(
    dataset, 
    [tam_treino, tam_val], 
    generator=torch.Generator().manual_seed(42)
)

# Dataloader para treino
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers= True,  # Manter workers persistentes se num_workers > 0
    drop_last=True
)

# Dataloader para validação
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,  # Manter workers persistentes se num_workers > 0
    drop_last=False
)

# Agrupar dataloaders
dataloaders = {
    "train": train_loader,
    "val": val_loader
}


#%%------------------------------------------------------------------------------------
# MODELO
#------------------------------------------------------------------------------------

# Instanciar modelo
if modelo == "EfficientUNet":
    from efficientUnet import EfficientUNet
    model = EfficientUNet(in_channels=n_canais, num_classes=n_classes, pretrained=pretreino).to(device)
elif modelo == "UNet":
    from unet import UNet
    model = UNet(in_channels=n_canais, out_channels=n_classes).to(device)
else:
    raise ValueError(f"Modelo '{modelo}' não reconhecido. Use 'EfficientUNet' ou 'UNet'.")

#%%------------------------------------------------------------------------------------
# OTIMIZADOR E FUNÇÃO DE PERDA
#------------------------------------------------------------------------------------
# Otimizador
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=taxa_aprendizagem,
    weight_decay=taxa_decaimento
)

#%%------------------------------------------------------------------------------------
# TREINAMENTO
#------------------------------------------------------------------------------------

if __name__ == '__main__':
    #import torch.multiprocessing as mp
    #mp.set_start_method('spawn', force=True)


    modelo_treinado = funcao_treino.treino(
        model           = model,
        dataloaders     = dataloaders,
        optimizer       = optimizer,
        #criterion      = WeightedIoULoss(class_weights),
        criterion       = losses.FocalTverskyLoss(),
        n_epochs        = epocas,
        device          = device,
        patience        = 20,
        use_amp         = True,
        hyperparametros = hiperparametros,
        verbose         = True,
    )
5