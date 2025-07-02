#%%-----------------------------------------------------------------------------------------------
# BIBLIOTECAS
#-------------------------------------------------------------------------------------------------
from matplotlib import cm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

from tqdm import tqdm as tqdm_inner  # Optional alias to distinguish inner bar
import os
import datetime
import numpy as np
import time
import copy
import pandas as pd

from sklearn.metrics import confusion_matrix

#%%-----------------------------------------------------------------------------------------------
# TREINAMENTO
#-------------------------------------------------------------------------------------------------
class_weights = [1/35.25, 1/58.43, 1/4.43, 1/0.01, 1/0.55, 1/1.32, 1/0.01]
class_weights = [w / sum(class_weights) for w in class_weights]

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
    hyperparametros:dict=None,  # Dicionário com hiperparâmetros adicionais
    n_classes:int=7,  # Número de classes para mIoU e acurácia
    save_dir=f"outputs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"  # Diretório para salvar o modelo e logs,
):
    
    # Verifica se o diretório de saída existe, caso contrário, cria
    os.makedirs(save_dir, exist_ok=True)

    # Salva hiperparâmetros em um arquivo JSON
    if hyperparametros is not None:
        import json
        with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparametros, f, indent=4)
        print(f"📄 Hiperparâmetros salvos em 'hyperparameters.json'.")

    # Scheduler: reduz a LR se a validação parar de melhorar
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=n_epochs * len(dataloaders['train']),
        pct_start=0.1,
        div_factor=25,
    final_div_factor=1e4
)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    # Inicializar histórico
    history = []


    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)

    for epoch in range(n_epochs):
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
            running_miou = 0.0

            # Iterar sobre os dados
            
            loop = tqdm_inner(dataloaders[phase], desc=f"{phase.capitalize()} [{epoch+1}/{n_epochs}]", leave=False)
            for inputs, labels in loop:
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
                        scaled_loss = scaler.scale(loss)
                        scaled_loss.backward()

                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_acc  += acc * inputs.size(0)
                running_miou  += mIoU * inputs.size(0)
                loop.set_postfix(loss=loss.item(), acc=acc, mIoU=mIoU)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc  = running_acc  / len(dataloaders[phase].dataset)
            epoch_miou = running_miou  / len(dataloaders[phase].dataset)

            if verbose:
                print(f"📊 {phase.upper()} — Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | mIoU: {epoch_miou:.4f}")

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
                
                all_preds = []
                all_labels = []

                all_preds.append(outputs.argmax(dim=1).detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                # Flatten and concatenate predictions for confusion matrix
                preds_flat = np.concatenate([p.flatten() for p in all_preds])
                labels_flat = np.concatenate([l.flatten() for l in all_labels])
                cm = confusion_matrix(labels_flat, preds_flat, labels=range(n_classes))


                # Save confusion matrix as CSV
                cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(n_classes)],
                                        columns=[f"pred_{i}" for i in range(n_classes)])
                cm_df.to_csv(os.path.join(save_dir, f"confusion_matrix_epoch_{epoch + 1}.csv"))                

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
        
        salvar_logs(history, save_dir)
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
    #now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(save_dir, f'training_log.csv'), index=False)
    print("📄 Logs salvos em 'training_log.csv'.")




#%%-----------------------------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
#-------------------------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedIoULoss(nn.Module):
    def __init__(self, weights, smooth=1.0):
        super().__init__()
        self.weights = torch.tensor(weights).float()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        if probs.device != self.weights.device:
            self.weights = self.weights.to(probs.device)

        ious = []
        for c in range(num_classes):
            pred_c = probs[:, c]
            true_c = targets_one_hot[:, c]
            inter = torch.sum(pred_c * true_c)
            union = torch.sum(pred_c + true_c - pred_c * true_c)
            iou = (inter + self.smooth) / (union + self.smooth)
            ious.append(iou)

        ious = torch.stack(ious)
        return 1 - torch.sum(ious * self.weights)


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights, smooth=1.0):
        super().__init__()
        self.weights = torch.tensor(weights).float()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        if probs.device != self.weights.device:
            self.weights = self.weights.to(probs.device)

        dices = []
        for c in range(num_classes):
            pred_c = probs[:, c]
            true_c = targets_one_hot[:, c]
            inter = torch.sum(pred_c * true_c)
            union = torch.sum(pred_c + true_c)
            dice = (2 * inter + self.smooth) / (union + self.smooth)
            dices.append(dice)

        dices = torch.stack(dices)
        return 1 - torch.sum(dices * self.weights)


class CombinedIoUDiceLoss(nn.Module):
    def __init__(self, weights, alpha=0.5):
        super().__init__()
        self.iou = WeightedIoULoss(weights)
        self.dice = WeightedDiceLoss(weights)
        self.alpha = alpha  # balance between Dice and IoU

    def forward(self, logits, targets):
        return self.alpha * self.dice(logits, targets) + (1 - self.alpha) * self.iou(logits, targets)




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

