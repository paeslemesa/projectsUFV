# Semantic Segmentation Model of MSI/Sentinel2 dataset CerraData4m Using Efficient-Unet and U-net

This study adapted Efficient-Unet model to satel-lite imagery from MultiSpectral Imager (MSI) sensor onboard Sentinel-2 constelation, comparing the same model with and with-out pretrained weights. The used dataset was Cerradata4mm, based on  TerraClass project. This study proved that using pretrained weights performs better, generally, than without using them. Another results was the variation of imbalanced classes in the metrics in the models.

---
## 1. How to use

First, these are the file names with their respective descriptions:

- <code>CNN_SensoriamentoRemoto.py</code> is the main code that receives all other codes and runs the training.
- <code>efficientUnet.py</code> is the architecture for the EfficientUnet B3 model adapted for the input dataset.
- <code>unet.py</code> is the architecture for Unet model adapted for the input dataset
- <code>CerraDataDataset.py</code> is the dataset code to adjust it to input into PyTorch
- <code>funcao_treino.py</code> is the training function, with the for loops for each epoch and mini-batches are computed
- <code>losses.py</code> is where the losses functions are described
- <code>output_analyses.ipynb</code>> generates the models summary and their graphs; this code also provides charts for model evaluation
- <predicao.py> predicts the model for 3 random imagens in the validation set and prints them

---
### 1.1. Downloading dataset

The used dataset is based on [Miranda et al (2025)](https://arxiv.org/abs/2502.00083) paper entitled *CerraData-4MM - A Multimodal Dataset on Cerrado for Land Use and Land Cover Classification*. <br>
The dataset can be downloaded via Kaggle: [CerraData-4MM](https://www.kaggle.com/datasets/cerranet/cerradata-4mm)

### 1.2. Adjusting path and hyperparameters
In <code>CNN_SensoriamentoRemoto.py</code>, you can change the path and hyperparameters that it will all be automatically adjusted from lines **33** to **57**.

```
# Caminho dos dados
caminho = "./cerradata_4mm/"

# Hiperparâmetros
batch_size       = 64
msi_bands        = [3,2,1,4] # The order of the bands fro MSI/Sentinel-2
num_workers      = 10  # or 4, or 8 based on your system
n_classes        = 7 
veg_index        = True # To use NDVI + EVI2 + SAVI
sar_bands        = True # To use VV+VH bands from Sentinel-1 C-band  
height, width    = 128, 128
epocas           = 100
taxa_aprendizagem= 1e-4 
taxa_decaimento  = 1e-3 
n_samples        = None # None para usar todo o dataset
transforms       = True
pretreino        = True # defines if the training will (True) or not (False) use pretrained weights (Imagenet1k)
modelo           = "UNet" # EfficientUNet or UNet
```

Run the model and wait until it finishes.

This is the training process:

![Training process flowchart](H:\Meu Drive\2025-1\INF692\TrabalhoFinal\fluxo_treino.drawio)

---
### 1.2.Analysing results

After the training phase is completed, you can analyse the results in the notebook <code>output_analysis.ipynb</code>.
There, you can print the model summary as a TXT file and as a flowchart. You can also plot **Loss** and **Validation metrics** per epoch, which are automatically saved to the output directory.

---
### 1.3. Predictions
In <code>predicao.ipynb</code>, you can run the prediction for 3 random images from the validation set from the <code>best_model.pth</code> in the selected output directory.



