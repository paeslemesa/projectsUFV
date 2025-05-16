"""
Fixed CerraDataset class with workaround for numpy array type issue
"""
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import tifffile as tiff
import rasterio

class CerraDataset(Dataset):
    def __init__(self, cam_dir: str, dispositivo: str = 'cpu', normalizacao: str = '0a1', transformar: bool = False):
        self.cam_dir = cam_dir
        self.normalizacao = normalizacao
        self.dispositivo = torch.device(dispositivo)
        self.transformar = transformar

        # Transformações de dados
        self.transformacoes = transforms.Compose([
            transforms.RandomRotation(45),     # Rotação aleatória
            transforms.RandomHorizontalFlip(p=0.5), # Espelhamento horizontal
            transforms.RandomVerticalFlip(p=0.5),   # Espelhamento vertical
        ]) if transformar else None

        # Carregar caminhos das imagens e máscaras
        self.opt_lista     = sorted(glob.glob(os.path.join(cam_dir, 'msi_images', '*.tif')))
        self.mascara_lista = sorted(glob.glob(os.path.join(cam_dir, 'semantic_7c', '*.tif')))

        if len(self.opt_lista) != len(self.mascara_lista):
            raise ValueError("Numero de imagens não casam com número de máscaras.")

    def _ler_imagem(self, caminho: str) -> np.ndarray:
        """Ler imagemTIFF usando GDAL"""
        with rasterio.open(caminho) as src:
            # Ler a imagem como um array numpy
            imagem = src.read()
            # Verifica se a imagem foi lida corretamente
            if imagem is None:
                raise ValueError(f"Não foi possível ler a imagem: {caminho}")
            return imagem


    def _normalizar(self, imagem: np.ndarray) -> np.ndarray:
        """Normalizar as imagens"""
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

    def __len__(self):
        """Retorna o número de imagens no dataset"""
        return len(self.opt_lista)

    def __getitem__(self, idx):
        # 1. Carregar e normalizar a imagem
        img_data = self._ler_imagem(self.opt_lista[idx])
        img_data = self._normalizar(img_data)
        
        # Converte para numpy array se necessário
        if not isinstance(img_data, np.ndarray):
            img_data = np.array(img_data)

        imagem = img_data.astype(np.float32)

        # Aplicar transformações se necessário
        if self.transformar and self.transformacoes:
            # Transformações de dados
            # Converte a imagem para tensor
            imagem = torch.tensor(imagem, dtype=torch.float32)
            imagem = self.transformacoes(imagem)


        
        # 2. Carregar a máscara
        mask_data = tiff.imread(self.mascara_lista[idx])
        mascara = torch.tensor(mask_data, dtype=torch.long)
        
        return imagem.to(self.dispositivo), mascara.to(self.dispositivo)