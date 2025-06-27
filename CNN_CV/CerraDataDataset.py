import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import rasterio
import albumentations as A
import random

class CerraDataset(Dataset):
    def __init__(
        self,
        cam_dir: str,
        normalizacao: str = '0a1',
        transformar: bool = False,
        dispositivo: str = 'cpu',
        seed: int = 42,
        red_idx: int = 3,
        nir_idx: int = 7,
        savi_l: float = 0.5,
    ):
        """
        Dataset com EVI2, NDVI e SAVI como bandas extras.
        """
        self.cam_dir = cam_dir
        self.normalizacao = normalizacao
        self.transformar = transformar
        self.dispositivo = dispositivo
        self.seed = seed
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.savi_l = savi_l

        # Semente para reprodutibilidade
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Caminhos de imagens e máscaras
        self.opt_lista = sorted(glob.glob(os.path.join(cam_dir, 'msi_images', '*.tif')))
        self.mascara_lista = sorted(glob.glob(os.path.join(cam_dir, 'semantic_7c', '*.tif')))

        if len(self.opt_lista) != len(self.mascara_lista):
            raise ValueError("Número de imagens não casa com número de máscaras.")

        self.transformacoes = A.Compose([
            A.Rotate(limit=45, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomCrop(height=128, width=128, p=0.5),
        ]) if transformar else None

    def __len__(self):
        return len(self.opt_lista)

    def _ler_imagem(self, caminho: str) -> np.ndarray:
        with rasterio.open(caminho) as src:
            return src.read()  # [C, H, W]

    def _ler_mascara(self, caminho: str) -> np.ndarray:
        with rasterio.open(caminho) as src:
            return src.read(1)  # [H, W]

    def _normalizar(self, imagem: np.ndarray) -> np.ndarray:
        imagem = np.clip(imagem, 1e-6, None)
        normalizada = np.zeros_like(imagem, dtype=np.float32)

        if self.normalizacao == '0a1':
            for j in range(imagem.shape[0]):
                min_val, max_val = np.min(imagem[j]), np.max(imagem[j])
                normalizada[j] = (imagem[j] - min_val) / (max_val - min_val + 1e-6)
        elif self.normalizacao == '1a1':
            for j in range(imagem.shape[0]):
                mean, std = np.mean(imagem[j]), np.std(imagem[j])
                normalizada[j] = (imagem[j] - mean) / (std + 1e-6)
        else:
            raise ValueError("Tipo de normalização inválida. Use '0a1' ou '1a1'.")
        
        return normalizada

    def _calcular_indices(self, imagem: np.ndarray) -> np.ndarray:
        """
        Calcula EVI2, NDVI e SAVI e retorna como 3 bandas empilhadas.
        """
        nir = imagem[self.nir_idx].astype(np.float32)
        red = imagem[self.red_idx].astype(np.float32)

        # NDVI
        ndvi = (nir - red) / (nir + red + 1e-6)

        # SAVI
        savi = (1 + self.savi_l) * (nir - red) / (nir + red + self.savi_l + 1e-6)

        # EVI2
        evi2 = 2.5 * (nir - red) / (nir + 2.4 * red + 1 + 1e-6)

        # [3, H, W]
        return np.stack([ndvi, savi, evi2], axis=0)

    def __getitem__(self, idx):
        # 1. Carrega imagem e máscara
        img = self._ler_imagem(self.opt_lista[idx])  # [C, H, W]
        mask = self._ler_mascara(self.mascara_lista[idx])  # [H, W]

        # 2. Calcula índices espectrais e adiciona como novas bandas
        indices = self._calcular_indices(img)  # [3, H, W]
        img = np.concatenate([img, indices], axis=0)  # [C+3, H, W]

        # 3. Normaliza imagem
        img = self._normalizar(img)

        # 4. Transforma para [H, W, C] para albumentations
        img = np.transpose(img, (1, 2, 0))  # [H, W, C]

        # 5. Aplica transformações
        if self.transformar and self.transformacoes:
            augmented = self.transformacoes(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        # 6. Volta para [C, H, W] e converte para tensor
        img = np.transpose(img, (2, 0, 1))  # [C, H, W]
        img_tensor = torch.tensor(img, dtype=torch.float32).to(self.dispositivo)
        mask_tensor = torch.tensor(mask, dtype=torch.long).to(self.dispositivo)

        return img_tensor, mask_tensor
