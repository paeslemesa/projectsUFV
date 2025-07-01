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
from DeepLabV3 import deeplabv3_Sentinel3
from efficientUnet import EfficientUNet

# Funções auxiliares
import funcao_treino


#%%------------------------------------------------------------------------------------
# ENTRADA DE DADOS
#------------------------------------------------------------------------------------

# Caminho dos dados
caminho = "/home/sabrina/Documents/Datasets/cerradata_4mm/"

# Hiperparâmetros
batch_size       = 64
num_workers      = os.cpu_count() // 2  # or 4, or 8 based on your system
n_classes        = 7   
n_canais         = 4
height, width    = 128, 128
epocas           = 20
taxa_aprendizagem= 1e-4 
taxa_decaimento  = 1e-3 
n_samples        = None # None para usar todo o dataset
transforms       = True

# Verificar se há GPU disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


#%%------------------------------------------------------------------------------------
# DATASETS E DATALOADERS
#------------------------------------------------------------------------------------

# Carregar dataset
dataset = CerraDataset(
    cam_dir=caminho, 
    dispositivo=device, 
    normalizacao='0a1', 
    transformar=transforms
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
    #num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=True
)

# Dataloader para validação
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    #num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
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
#model = deeplabv3_Sentinel3(num_classes=n_classes, in_channels=n_canais).to(device)
model = EfficientUNet(in_channels=n_canais, num_classes=n_classes, pretrained=True).to(device)


#%%------------------------------------------------------------------------------------
# OTIMIZADOR E FUNÇÃO DE PERDA
#------------------------------------------------------------------------------------

# Otimizador
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=taxa_aprendizagem,
    weight_decay=taxa_decaimento
)

# Funções de perda
class SmoothCrossEntropyLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        nll_loss = -log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Use SmoothCrossEntropy + Dice
ce_loss = SmoothCrossEntropyLoss(smoothing=0.1)
dice_loss = funcao_treino.DiceLoss()

def perdas_combinadas(logits, targets, alpha=0.7):
    return alpha * ce_loss(logits, targets) + (1 - alpha) * dice_loss(logits, targets)



#%%------------------------------------------------------------------------------------
# TREINAMENTO
#------------------------------------------------------------------------------------

if __name__ == '__main__':
    #import torch.multiprocessing as mp
    #mp.set_start_method('spawn', force=True)


    modelo_treinado = funcao_treino.treino(
        model      = model,
        dataloaders= dataloaders,
        optimizer  = optimizer,
        criterion  = perdas_combinadas,
        n_epochs   = epocas,
        device     = device,
        patience   = 20,
        use_amp    = False,
        verbose    = True,
    )
5