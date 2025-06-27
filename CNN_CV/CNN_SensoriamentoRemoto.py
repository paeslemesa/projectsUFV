#%%------------------------------------------------------------------------------------
# BIBLIOTECAS
#------------------------------------------------------------------------------------
import os

# PyTorch
import torch
from torch.utils.data import random_split, DataLoader, Subset
import torch.multiprocessing as mp

# Visualização de modelo
from torchinfo import summary
from torchview import draw_graph

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
caminho = "/mnt/d/Doutorado_UFV/DATASETS/cerradata4mm/cerradata_4mm"

# Hiperparâmetros
batch_size       = 16       
num_workers      = 4        
n_classes        = 7        
n_canais         = 12       
height, width    = 128, 128

epocas           = 1    
taxa_aprendizagem= 1e-2 
taxa_decaimento  = 1e-2 

# Verificar se há GPU disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Usando dispositivo: {device}")


#%%------------------------------------------------------------------------------------
# DATASETS E DATALOADERS
#------------------------------------------------------------------------------------

# Carregar dataset
dataset = CerraDataset(
    cam_dir=caminho, 
    dispositivo=device, 
    normalizacao='0a1', 
    transformar=True
)

if dataset is None or len(dataset) == 0:
    raise ValueError("Dataset não carregado corretamente ou está vazio.")


#subset_indices = list(range(1000))
#subset = Subset(dataset, subset_indices)
#dataset = subset

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
    #pin_memory=torch.cuda.is_available()
)

# Dataloader para validação
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,

    #pin_memory=torch.cuda.is_available()
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
model = EfficientUNet(in_channels=n_canais, num_classes=n_classes)

# Visualizar arquitetura do modelo
input_tensor = torch.randn(batch_size, n_canais, height, width).to(device)
graph = draw_graph(model, input_data=input_tensor, depth=3)
graph.visual_graph.render("segformer_arch", format="png")

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
ce_loss   = torch.nn.CrossEntropyLoss()
dice_loss = funcao_treino.DiceLoss()

def perdas_combinadas(logits, targets):
    """Média das perdas CrossEntropy e DiceLoss."""
    return (ce_loss(logits, targets) + dice_loss(logits, targets)) / 2


#%%------------------------------------------------------------------------------------
# TREINAMENTO
#------------------------------------------------------------------------------------

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)


    modelo_treinado = funcao_treino.treino(
        model      = model,
        dataloaders= dataloaders,
        optimizer  = optimizer,
        criterion  = ce_loss,
        n_epochs   = epocas,
        device     = device,
        patience   = 20,
        use_amp    = True,
        verbose    = True,
    )
