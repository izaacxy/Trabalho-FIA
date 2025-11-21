# -*- coding: utf-8 -*-
"""Tutorial_Helper_Functions - PyTorch Version

Este arquivo contém a definição da CNN e funções auxiliares para o projeto LTN.
"""

import numpy as np
import pandas as pd
import os
import pickle
import cv2
import random 
import seaborn as sns
import logging; logging.basicConfig(level=logging.INFO)  
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imread
from itertools import product 
from collections import defaultdict 
from tqdm import tqdm 
import ltn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.io import imshow

# Dicionário de mapeamento de classes para cores (essencial para o LTN Clevr)
dict_nb_to_colours= {0: 'dark blue', 1: 'green', 2: 'red',  3: 'baby blue',  4: 'grey', 5: 'light blue'}

# -------------------------------------------------------------------------------
# Funções de resumo e visualização (para ajudar na análise dos dados)
def summarize_imported_dataset_on_object_level(d):     
  print("Há",len(d), "exemplos de treinamento")
  print("com as seguintes informações:",list(d[0].keys()))
  x = ['object_image'] 
  y = ["color","shape"] 
  print("Para o atributo 'color', há",len(set([x["color"] for x in d])), "valores possíveis")
  print("Para o atributo 'shape', há",len(set([x["shape"] for x in d])), "valores possíveis")

def visualize_example(example):
  fig,axs = plt.subplots(1,3,figsize=(10,3))
  axs[0].imshow(cv2.cvtColor(example[0].object_image, cv2.COLOR_BGR2RGB))
  axs[0].set_title(f"Objeto 1: {example[0].color}, {example[0].shape}")
  axs[1].imshow(cv2.cvtColor(example[1].object_image, cv2.COLOR_BGR2RGB))
  axs[1].set_title(f"Objeto 2: {example[1].color}, {example[1].shape}")
  
  # Visualiza a cena inteira (se disponível)
  if hasattr(example[0], 'scene_image'):
    axs[2].imshow(cv2.cvtColor(example[0].scene_image, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Cena Completa")
  else:
    axs[2].axis('off')

  plt.show()

# -------------------------------------------------------------------------------
# Definição do Modelo Neural (CNN_detector) que serve como Predicado no LTN
class CNN_detector(nn.Module):
    """
    Define a CNN que irá aprender a classificar atributos (cor, forma) dos objetos.
    Ela transforma a imagem de um objeto em um vetor de probabilidades/verdades.
    """
    def __init__(self, n_classes):
        super(CNN_detector, self).__init__()
        # Camadas convolucionais (extração de features da imagem)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Camadas classificadoras (mapeia features para classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x, class_label=None):
        # 1. TRATAMENTO LTN: Se o input for um LTNObject, usa seu valor (tensor)
        if hasattr(x, 'value'):
            x = x.value
            
        # 2. Processamento CNN
        features = self.conv_layers(x)
        logits = self.classifier(features)
        
        # 3. TRATAMENTO DE SAÍDA: Se a label da classe for fornecida, seleciona a probabilidade daquela classe
        if class_label is not None:
            # 3a. Trata a label se for um LTNObject
            if hasattr(class_label, 'value'):
                class_label = class_label.value
                
            # Garante que class_label é um índice inteiro
            if class_label.dtype == torch.float32:
                class_label = class_label.long()
                
            # Seleciona o logit correspondente à classe fornecida
            selected_logits = logits[torch.arange(logits.shape[0]), class_label]
            
            # Aplica sigmoid para obter a probabilidade no range [0,1] (Grau de Verdade)
            return torch.sigmoid(selected_logits)
        
        # Retorna todos os logits se nenhuma classe específica for solicitada
        return logits

# -------------------------------------------------------------------------------
# CLASSE DE DATASET: Para carregar os dados de forma eficiente no PyTorch
class CLEVR_Dataset(Dataset):
    """Dataset personalizado para o carregamento dos dados CLEVR simplificados."""
    
    def __init__(self, objects, classes_to_nb, attributes, is_train=True):
        self.objects = objects
        self.classes_to_nb = classes_to_nb
        self.attributes = attributes # 'color' ou 'shape'
        
        # Apenas um snippet para mostrar a estrutura, a lógica de carregamento real
        # dependeria de um arquivo .pkl ou .json que você deve ter recebido no zip.
        
    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        obj = self.objects[idx]
        
        # Converte a imagem (que deve estar em formato adequado, ex: HWC para CHW)
        image = obj['object_image']
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Converte o atributo (cor/forma) para seu índice numérico
        attribute_label = self.classes_to_nb[obj[self.attributes]]
        attribute_tensor = torch.tensor(attribute_label, dtype=torch.long)
        
        return image, attribute_tensor

# Outras funções (como carregar os dados reais, 'load_dataset') seriam incluídas aqui...
# -------------------------------------------------------------------------------
