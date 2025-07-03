from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
import numpy as np

def make_sampler(dataset, n_classes):
    # 1) Count total pixels per class (global freq)
    class_counts = np.zeros(n_classes, dtype=float)
    for _, mask in dataset:
        vals, counts = np.unique(mask.numpy(), return_counts=True)
        class_counts[vals.astype(int)] += counts
    class_freqs = class_counts / class_counts.sum()
    
    # 2) Inverse‐sqrt weighting (so extremes aren’t too huge)
    inv_sqrt = 1.0 / np.sqrt(class_freqs + 1e-6)
    class_weights = inv_sqrt / inv_sqrt.sum()

    # 3) Per‐sample weight = average weight of the classes present in that mask
    sample_weights = []
    for _, mask in dataset:
        mask = mask.numpy().flatten().astype(int)
        # find unique classes in this mask
        uniq = np.unique(mask)
        # weight = mean of their class_weights
        sample_weights.append(class_weights[uniq].mean())

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

# Usage when building train_loader:
#train_sampler = make_sampler(train_dataset, n_classes=7)
#train_loader = DataLoader(
#    train_dataset,
#    batch_size=batch_size,
#    sampler=train_sampler,
#    pin_memory=True,
#    drop_last=True,
#    num_workers=num_workers
#)
