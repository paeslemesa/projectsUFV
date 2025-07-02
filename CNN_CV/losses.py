import torch
import torch.nn.functional as F
import numpy as np


#-------------------------------------------------------------------------
# Pesos de classe
# Pesos de classe para lidar com desbalanceamento de classes
# Aqui usamos uma abordagem simples de "sqrt" inversa da frequência
#--------------------------------------------------------------------------
raw = np.array([35.25,58.43,4.43,0.01,0.55,1.32,0.01]) # Frequências de classes
inv_sqrt = 1 / np.sqrt(raw) # Inverso da raiz quadrada das frequências
class_weights = inv_sqrt / inv_sqrt.sum() # Normalizar os pesos para que a soma seja 1
# Normalizar pesos para o intervalo [0, 1]
class_weights = class_weights / class_weights.max()

#-------------------------------------------------------------------------
# IoU e Dice Losses
#-------------------------------------------------------------------------
class WeightedIoULoss(torch.nn.Module):
    def __init__(self, weights, smooth=1.0):
        super().__init__()
        self.weights = torch.tensor(weights).float()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
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


class WeightedDiceLoss(torch.nn.Module):
    def __init__(self, weights, smooth=1.0):
        super().__init__()
        self.weights = torch.tensor(weights).float()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
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

#-------------------------------------------------------------------------
# Tversky Loss
#-------------------------------------------------------------------------
class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.size(1)
        t = torch.nn.functional.one_hot(targets, num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        TP = torch.sum(probs * t, dims)
        FP = torch.sum(probs * (1 - t), dims)
        FN = torch.sum((1 - probs) * t, dims)
        tversky = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)
        return 1 - tversky.mean()



class CombinedIoUDiceLoss(torch.nn.Module):
    def __init__(self, weights, alpha=0.5):
        super().__init__()
        self.iou = WeightedIoULoss(weights)
        self.dice = WeightedDiceLoss(weights)
        self.alpha = alpha  # balance between Dice and IoU

    def forward(self, logits, targets):
        return self.alpha * self.dice(logits, targets) + (1 - self.alpha) * self.iou(logits, targets)


#-------------------------------------------------------------------------
# Lovasz-Softmax Loss
# Adapted from https://github.com/bermanmaxim/LovaszSoftmax
#-------------------------------------------------------------------------
def lovasz_softmax_flat(probs, labels, classes='all', per_image=False, ignore=None):
    # Implementation adapted from https://github.com/bermanmaxim/LovaszSoftmax
    if per_image:
        loss = torch.mean(torch.stack([
            lovasz_softmax_flat(*flatten_probas(p.unsqueeze(0), l.unsqueeze(0), ignore), classes=classes)
            for p, l in zip(probs, labels)
        ]))
        return loss
    C = probs.size(1)
    losses = []
    for c in range(C if classes == 'all' else classes):
        fg = (labels == c).float()          # foreground for class c
        if fg.sum() == 0:
            continue
        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    return torch.mean(torch.stack(losses))

def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jacc = 1. - intersection / union
    jacc[1:] = jacc[1:] - jacc[:-1]
    return jacc

def flatten_probas(probs, labels, ignore=None):
    # Flattens predictions in the batch
    B, C, H, W = probs.size()
    probs = probs.permute(0,2,3,1).reshape(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probs, labels
    keep = labels != ignore
    return probs[keep], labels[keep]

class LovaszSoftmaxLoss(torch.nn.Module):
    def __init__(self, per_image=False, ignore_index=None):
        super().__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        return lovasz_softmax_flat(probs, labels, per_image=self.per_image, ignore=self.ignore_index)


#-------------------------------------------------------------------------
# Focal Tversky Loss
#-------------------------------------------------------------------------
class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-6):
        """
        Focal Tversky Loss for semantic segmentation.

        Args:
            alpha (float): weight for false negatives.
            beta  (float): weight for false positives.
            gamma (float): focusing parameter (like focal loss).
            smooth (float): small constant to avoid division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W) raw model outputs
        targets: (B, H, W)  integer class labels
        """
        # probabilities
        probs = F.softmax(logits, dim=1)
        C = logits.size(1)

        # one-hot encode targets
        t = F.one_hot(targets, C).permute(0,3,1,2).float()  # (B, C, H, W)

        dims = (0,2,3)  # sum over batch, height, width

        # True positives, false negatives, false positives per class
        TP = torch.sum(probs * t, dims)
        FN = torch.sum((1 - probs) * t, dims)
        FP = torch.sum(probs * (1 - t), dims)

        # Tversky index per class
        tversky = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)
        # Focal Tversky loss
        loss = torch.pow((1 - tversky), self.gamma)

        # average over classes
        return loss.mean()
