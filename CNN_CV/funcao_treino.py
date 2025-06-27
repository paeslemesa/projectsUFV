#%%-----------------------------------------------------------------------------------------------
# BIBLIOTECAS
#-------------------------------------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

from tqdm import tqdm
import os
import datetime
import numpy as np
import time
import copy
import pandas as pd

#%%-----------------------------------------------------------------------------------------------
# TREINAMENTO
#-------------------------------------------------------------------------------------------------


class EarlyStopping:
    def __init__(self, patience=15, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.best_epoch = 0
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, epoch_loss, epoch):
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"✅ New best loss: {epoch_loss:.4f} at epoch {epoch+1}")
            return True  # Indicates improvement
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ No improvement. EarlyStopping counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"⛔ Early stopping triggered at epoch {epoch+1}")
            return False




def treino(
    model:nn.Module,
    dataloaders:dict,
    optimizer:torch.optim.Optimizer,
    criterion:nn.Module,
    n_epochs:int=10,
    device:str="cpu",
    patience:int=5,
    use_amp:bool=True,
    verbose:bool=True,
    save_dir=f"outputs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"  # Diretório para salvar o modelo e logs,
):
    os.makedirs(save_dir, exist_ok=True)

    # Scheduler: reduz a LR se a validação parar de melhorar
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5 )  # Variáveis para Early Stopping
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    # Inicializar histórico
    history = []

    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)

    for epoch in tqdm(range(n_epochs)):
        since = time.time()
        if verbose:
            print(f"\n🟦 Época {epoch+1}/{n_epochs}")
            print("-" * 50)

        # Cada época tem uma fase de treino e uma de validação
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_acc = 0.0
            runnin_miou = 0.0

            # Iterar sobre os dados
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    device_name = "cuda" if torch.cuda.is_available() else "cpu"
                    with torch.amp.autocast(device_type = device_name, enabled=use_amp):
                        outputs = model(inputs)
                        #print("Output:", outputs.shape)
                        #print("Target:", labels.shape)
                        loss = criterion(outputs, labels)
                        acc  = ValidationMetrics().pixel_accuracy(outputs.argmax(dim=1), labels)
                        mIoU = ValidationMetrics().compute_mean_iou(outputs.argmax(dim=1), labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_acc  += acc * inputs.size(0)
                runnin_miou  += mIoU * inputs.size(0)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc  = running_acc  / len(dataloaders[phase].dataset)
            epoch_miou = runnin_miou  / len(dataloaders[phase].dataset)

            if verbose:
                print(f"{phase} Loss: {epoch_loss:.4f}")

            # Histórico
            history.append({
                'epoch': epoch + 1,
                'phase': phase,
                'loss': epoch_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'tempo': (time.time() - since),
                'tempo': str(datetime.timedelta(seconds=int(time.time() - since))),
                'accuracy': epoch_acc, 
                'mIoU': epoch_miou  
            })

            # Fase de validação — verificar Early Stopping e Checkpoint
            if phase == 'val':
                scheduler.step(epoch_loss)

                if epoch_loss < best_loss:
                    epochs_no_improve = 0
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                    epochs_no_improve = 0
                    if verbose:
                        print("✅ Melhor modelo salvo.")
                else:
                    epochs_no_improve += 1
                    if verbose:
                        print(f"⏳ Sem melhoria por {epochs_no_improve} épocas.")

                if epochs_no_improve >= patience:
                    print("🛑 Early stopping ativado.")
                    model.load_state_dict(best_model_wts)
                    salvar_logs(history, save_dir)
                    return model

        time_elapsed = time.time() - since
        if verbose:
            print(f"⏱ Tempo da época: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    print("\n🏁 Treinamento concluído.")
    model.load_state_dict(best_model_wts)
    salvar_logs(history, save_dir)
    return model


def salvar_logs(history, save_dir):
    """Salva o histórico de treino em CSV."""
    df = pd.DataFrame(history)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(save_dir, f'training_log_{now}.csv'), index=False)
    print("📄 Logs salvos em 'training_log.csv'.")




#%%-----------------------------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
#-------------------------------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """
    Função de perda Dice Loss para segmentação semântica.
    A Dice Loss é uma função de perda que mede a similaridade entre duas amostras.
    Ela é frequentemente usada em tarefas de segmentação semântica, onde o objetivo é prever a máscara de um objeto em uma imagem.
    A Dice Loss é calculada como 1 menos o coeficiente de Dice, que é definido como:
        Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
    onde |X| e |Y| são os tamanhos das duas amostras (neste caso, a previsão e a máscara real) e |X ∩ Y| é o tamanho da interseção entre as duas amostras.
    A Dice Loss é uma função de perda não negativa, onde 0 indica uma previsão perfeita.
    """
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Calcula a Dice Loss entre as previsões e os rótulos reais.
        Args:
            logits (torch.Tensor): Tensor de previsões do modelo.
            targets (torch.Tensor): Tensor de rótulos reais.
            num_classes (int): Número de classes.
        Returns:
            torch.Tensor: Valor da Dice Loss.
        """
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        intersection = torch.sum(probs * targets_one_hot)
        union = torch.sum(probs + targets_one_hot)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
    
class ValidationMetrics(object):
    """
    Classe para calcular métricas de validação durante o treinamento.
    """
    def __init__(self):
        pass
    
    def pixel_accuracy(self, preds:torch.Tensor, labels:torch.Tensor):
        """
        Calcula a acurácia de pixel.
        Args:
            preds (torch.Tensor): Tensor de previsões do modelo.
            labels (torch.Tensor): Tensor de rótulos reais.
        Returns:
            float: Acurácia de pixel.
        """
        correct = (preds == labels).float()
        acc = correct.sum() / correct.numel()
        return acc.item()
      
    def compute_mean_iou(self, preds:torch.Tensor, labels:torch.Tensor, n_classes:int=7):
        """
        Calcula o IoU médio (Intersection over Union) para cada classe.
        Args:
            preds (torch.Tensor): Tensor de previsões do modelo.
            labels (torch.Tensor): Tensor de rótulos reais.
            n_classes (int): Número de classes.
        Returns:
            float: IoU médio.
        """
        iou_list = []
        preds    = preds.view(-1)
        labels   = labels.view(-1)

        for cls in range(n_classes):
            pred_inds  = preds  == cls
            label_inds = labels == cls

            intersection = (pred_inds & label_inds).sum().float()
            union = (pred_inds | label_inds).sum().float()

            if union == 0:
                iou = torch.tensor(float('nan'))  # ignore this class in IoU
            else:
                iou = intersection / union

            iou_list.append(iou)

        # Return mean over all non-NaN classes
        iou_list = torch.tensor(iou_list)
        return torch.nanmean(iou_list).item()

