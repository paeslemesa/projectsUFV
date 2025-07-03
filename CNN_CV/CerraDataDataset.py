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
        bands: list = [3, 1, 2, 4],  # RGB + NIR
        veg_indexes = True, # Indica se deve calcular EVI2, NDVI e SAVI
        red_idx: int = 3,
        nir_idx: int = 4,
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
        self.bands = bands 
        self.veg_indexes = veg_indexes  # Indica se deve calcular EVI2, NDVI e SAVI
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.savi_l = savi_l

        # Semente para reprodutibilidade
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Caminhos de imagens e máscaras
        self.opt_lista = sorted(glob.glob(os.path.join(cam_dir, 'msi_images', '*.tif')))
        self.sar_lista = sorted(glob.glob(os.path.join(cam_dir, 'sar_images', '*.tif')))
        self.mascara_lista = sorted(glob.glob(os.path.join(cam_dir, 'semantic_7c', '*.tif')))

        if len(self.opt_lista) != len(self.mascara_lista):
            raise ValueError("Número de imagens não casa com número de máscaras.")

        self.transformacoes = A.Compose([
            #A.Rotate(limit=45, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            #A.RandomCrop(height=128, width=128, p=0.5),
        ]) if transformar else None

    def __len__(self):
        return len(self.opt_lista)

    def _ler_imagem(self, caminho: str) -> np.ndarray:
        with rasterio.open(caminho) as src:
            return src.read()  # [C, H, W]

    def _ler_mascara(self, caminho: str) -> np.ndarray:
        with rasterio.open(caminho) as src:
            return src.read(1)  # [H, W]

    def _normalizar(self, imagem: np.ndarray, modality: str) -> np.ndarray:
        imagem = np.clip(imagem, 1e-6, None)
        normalizada = np.zeros_like(imagem, dtype=np.float32)

        max, min, mean, stddev = self.data_info(modality)  # Obtém informações estatísticas para MSI
        if self.normalizacao == '0a1':
            for j in range(imagem.shape[0]):
                normalizada[j] = (imagem[j] - min[j]) / (max[j] - min[j])
        elif self.normalizacao == '1a1':
            for j in range(imagem.shape[0]):
                normalizada[j] = ((np.log(imagem[j]) - np.log(mean[j])) / np.log(stddev[j]))
        else:
            raise ValueError("Tipo de normalização inválida. Use '0a1' ou '1a1'.")
        
        return normalizada
    
    def data_info(self, modality):
        # Extracted from: https://github.com/ai4luc/CerraData-4MM/blob/main/CerraData-4MM%20Experiments/util/dataset_loader.py
        # SAR statistical information
        if modality == 'sar':
            min = [0.16844511032105, 0.18629205226898335]
            max = [1877.8493041992167, 1303.7864481607917]
            mean = [104.87897585598166, 95.52668271493417]
            stddev = [79.8024668186095, 63.256644370816836]

            return max, min, mean, stddev

        # MSI statistical information
        elif modality == 'msi':
            min = [99.78856658935547, 332.65665627643466, 347.161809168756, 331.4168453961611,
                196.89053159952164, 240.9765984416008, 261.34731489419937, 342.50664601475,
                277.87501442432404, 246.40860325098038, 265.9057685136795, 226.23770987987518]
            max = [7349.042938232482, 8987.99301147458, 8906.377044677738, 9027.435272216775,
                9090.25390625, 8949.610290527282, 8955.640045166012, 9491.945373535062,
                9026.07144165042, 11857.606872558594, 11817.384948730469, 13970.691894531188]
            mean = [1331.2999603920011, 1422.618248839035, 1648.7418838236356, 1811.0396095371318,
                    2243.6360604171587, 2862.469356914663, 3158.7246770243464, 3253.5804747400075,
                    3464.1887187200564, 3463.5260019211623, 3635.662557047575, 2740.6395025025904]
            stddev = [436.04697715189127, 484.32797096427566, 549.125419913045, 741.2668466992163,
                    788.8006282648606, 860.9668486457188, 963.2983618801512, 1000.2677835011111,
                    1087.111000434025, 1062.9960118331512, 1373.6088616321088, 1125.5168224477407]

            return max, min, mean, stddev

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
        img_optical = self._ler_imagem(self.opt_lista[idx])  # [C, H, W]
        img_sar = self._ler_imagem(self.sar_lista[idx])  # [C, H, W] 
        mask = self._ler_mascara(self.mascara_lista[idx])  # [H, W]  

        # 2. Calcula índices espectrais e adiciona como novas bandas
        indices = self._calcular_indices(img_optical)  # [3, H, W]
        
        # 3. Normaliza imagens
        img_optical = self._normalizar(img_optical, modality='msi')  # Normaliza MSI
        img_sar = self._normalizar(img_sar, modality='sar')  # Normaliza SAR

        # 4. Seleciona bandas específicas
        img_optical = img_optical[self.bands,:,:] # RGBNIR (3, 1, 2, 4 são os índices das bandas RGB+NIR)
        
        # 5. Empilha imagens ópticas e SAR
        img = np.concatenate([img_optical, img_sar], axis=0)  # [OPT + SAR, H, W]

        # 6. Adiciona índices espectrais se necessário
        if self.veg_indexes:
            img = np.concatenate([img, indices], axis=0)  # [C+3, H, W]

        # 7. Transforma para [H, W, C] para albumentations
        img = np.transpose(img, (1, 2, 0))  # [H, W, C]

        # 8. Aplica transformações
        if self.transformar and self.transformacoes:
            augmented = self.transformacoes(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        # 9. Volta para [C, H, W] e converte para tensor
        img = np.transpose(img, (2, 0, 1))  # [C, H, W]
        img_tensor = torch.tensor(img, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        return img_tensor, mask_tensor
