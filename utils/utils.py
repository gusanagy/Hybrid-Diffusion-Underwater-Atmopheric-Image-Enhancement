import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
import albumentations as A
import math
import os
import random
import math
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def list_images(directory):
    """
    Lists all images in the given directory with extensions .png or .jpg.

    :param directory: The directory to search for images.
    :return: A list of file paths to the images.
    """
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths
import random

def split_data(data_list, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, shuffle=False):
    """
    Divide uma lista em três partes: treino, validação e teste com proporções especificadas.

    Parâmetros:
        data_list (list): A lista de dados a ser dividida.
        train_ratio (float): Proporção para os dados de treino (padrão: 0.7).
        val_ratio (float): Proporção para os dados de validação (padrão: 0.1).
        test_ratio (float): Proporção para os dados de teste (padrão: 0.2).
        shuffle (bool): Se True, embaralha a lista antes de dividir (padrão: True).

    Retorno:
        tuple: Três listas contendo os dados de treino, validação e teste.
    """
    # Verifica se as proporções somam 1
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("As proporções de treino, validação e teste devem somar 1.")

    # Opcionalmente embaralha a lista
    if shuffle:
        random.shuffle(data_list)

    # Calcula os tamanhos das divisões
    total_len = len(data_list)
    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    test_size = total_len - train_size - val_size  # Garante que o restante vai para teste

    # Divide a lista
    train_data = data_list[:train_size]
    val_data = data_list[train_size:train_size + val_size]
    test_data = data_list[train_size + val_size:]

    return train_data, test_data, val_data

"""
fazer nova funcao para carregar os enderecos dos datasets e suas validacoes
Estou confundindo imagem pra treino, pra teste, para validacao e dado anotado

as imagens para teste e treino podem ter anoitação e devem seguir a proporcao de treino teste e validacao.

ja as imagens de validacao nao precisam ter anotacao, mas caso haja nao tem problema. A rede nunca ira ver as imagens de validacao
nem na rotina de treino nem de teste. Apenas na validacao final. 

Fazer o load dos dados anotados para as metricas posteriormente.


"""

def load_EUVP_paths(dataset_path = "data/EUVP",split=False): #errado Arrumar, usar apenas as imagens do treino A
     # Inicializa as listas para os caminhos das imagens
    train_img = []
    test_img = []
    val_img = []
    #data/EUVP/Paired/underwater_dark/trainA/264286_00007889.jpg
   # Define os padrões de busca para os diretórios
    trainA_pattern = glob.glob(os.path.join(dataset_path, "Paired", "*", "trainA", "*.jpg"))
    #trainB_pattern = os.path.join(dataset_path, "Paired", "*", "trainB", "*.jpg")
    #validation_pattern = os.path.join(dataset_path, "Paired", "*", "validation", "*.jpg")

    # # Faz a busca pelos caminhos das imagens
    # train_img.extend(glob.glob(trainA_pattern)) 
    # test_img.extend(glob.glob(trainB_pattern)) # Dados anotados e pareados
    # val_img.extend(glob.glob(validation_pattern)) # Imagens para Validacao sem dados pareados

  

    return split_data(trainA_pattern)

def load_HDR_paths(dataset_path="data/HDR+ Burst_20171106_subset", task="train", split=False):
    """
    Carrega os caminhos das imagens do dataset HDR+.

    Parâmetros:
        dataset_path (str): Caminho base para o dataset HDR+.
        task (str): Tarefa a ser executada (não utilizada diretamente neste código).
        split (bool): Indica se o split de treino/validação deve ser retornado.

    Retorna:
        tuple: Listas com caminhos para treino, teste e validação.
    """

    # Localiza todas as imagens no dataset
    imgs = []
    imgs.extend(glob.glob(os.path.join(dataset_path, "gallery_20171023", "*.jpg")))

    return split_data(imgs)

def load_HICRD_paths(dataset_path = "data/HICRD", task="train",split=False):
    train_img = []
    #test_img = []
    
    train_img.extend(glob.glob(os.path.join(dataset_path, "trainA_paired", "*.png")))
    #test_img.extend(glob.glob(os.path.join(dataset_path, "trainB_paired", "*.png")))

    #print(f"Total de imagens encontradas: {len(train_img) , len(test_img)}")

    return split_data(train_img)


def load_LSUI_paths(dataset_path = "data/LSUI", task="train",split=False):
    train_img = []
    #test_img = []
    
    train_img.extend(glob.glob(os.path.join(dataset_path, "input", "*.jpg")))

    return split_data(train_img)

def load_TM_DIED_paths(dataset_path = "data/TM-DIED", task="train",split=False):
    train_img = []
    #test_img = []
    
    train_img.extend(glob.glob(os.path.join(dataset_path, "*.jpg")))

    return split_data(train_img)


def load_UIEB_paths(dataset_path = "data/UIEB", task="train",split=False):
    train_img = []
    #test_img = []
    
    train_img.extend(glob.glob(os.path.join(dataset_path,"train", "*.png")))

    return split_data(train_img)

def load_RUIE_paths(dataset_path = "data/RUIE", task="train",split=False):
    train_img = []
    #test_img = []
    
    train_img.extend(glob.glob(os.path.join(dataset_path,"*","train", "*.jpg")))

    return split_data(train_img)

def load_image(image_path):
    """
    Carrega uma imagem de um caminho específico usando OpenCV.

    Args:
        image_path (str): Caminho para a imagem.

    Returns:
        np.ndarray: Imagem carregada no formato RGB.
    """
    # Carregar a imagem no formato BGR (padrão do OpenCV)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {image_path}")
    
    # Converter para o formato RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Retornar a imagem como um array numpy
    return image

    
    
class Atmospheric_Dataset(data.Dataset):
    def __init__(self, atmospheric_dataset_name: str, batch_size:int = 8, transforms=None, task: str = "train"):
        self.dataset_name = atmospheric_dataset_name
        self.batch_size = batch_size
        self.task =task

        # Define transformations
        if transforms is None:
            self.transform = A.Compose(
            [
                A.Resize(height=256, width=256),
                ToTensorV2(),
            ]
            )
        else:
            self.transform = transforms

        # Load dataset paths based on the dataset name
        self._load_datasets()
        
    def _load_datasets(self):

        #Atmospheric Dataset
        if self.dataset_name == "HDR+":
            self.train_img_a, self.test_img_a, self.val_img_a = load_HDR_paths()
        elif self.dataset_name == "TM-DIED":
            self.train_img_a, self.test_img_a, self.val_img_a = load_TM_DIED_paths()
        
                       
    def __len__(self):
        #Seleciona o tamanho do dataset a ser usado baseado na combinação escolhida
        if self.task == "train":
            return len(self.train_img_a)
        elif self.task == 'test':
            return len(self.test_img_a)
        else:
            return len(self.val_img_a)

    def __getitem__(self, idx):
        if self.task == "train":
            img_path_a = self.train_img_a[idx]
            img= self.transform(image=load_image(img_path_a))  # Implement `load_image` to load an image from a file path
        elif self.task == "test":
            img_path_a = self.test_img_a[idx]
            img= self.transform(image=load_image(img_path_a)) 
        else:
            img_path_a = self.val_img_a[idx]
            img= self.transform(image=load_image(img_path_a)) 
        return img["image"]

    
class Underwater_Dataset(data.Dataset):
    def __init__(self, underwater_dataset_name: str ,  transforms=None, task: str = "train"):
        self.underwater_dataset_name = underwater_dataset_name
        self.transform = transforms
        self.task =task

        # Define transformations
        if transforms is None:
            self.transform = A.Compose(
            [
                A.Resize(height=256, width=256),
                ToTensorV2(),
            ]
            )
        else:
            self.transform = transforms

        # Load dataset paths based on the dataset name
        #Underwater datasets
        if self.underwater_dataset_name == "HICRD":
            self.train_img_u, self.test_img_u, self.val_img_u = load_HICRD_paths()
        elif self.underwater_dataset_name == "LSUI":
            self.train_img_u, self.test_img_u, self.val_img_u = load_LSUI_paths()
        elif self.underwater_dataset_name == "UIEB":
            self.train_img_u, self.test_img_u, self.val_img_u = load_UIEB_paths()
        elif self.underwater_dataset_name == "RUIE":
            self.train_img_u, self.test_img_u, self.val_img_u = load_RUIE_paths()
        elif self.underwater_dataset_name == "EUVP":
            self.train_img_u, self.test_img_u, self.val_img_u = load_EUVP_paths()
        else:
            raise ValueError(
                f"Dataset {self.underwater_dataset_name} not found. Choose between 'EUVP', 'HDR+', 'HICRD', 'LSUI', 'TM-DIED', 'UIEB' or 'RUIE'"
            )

    def __len__(self):
        #Seleciona o tamanho do dataset a ser usado baseado na combinação escolhida
        if self.task == "train":
            return len(self.train_img_u)
        elif self.task == 'test':
            return len(self.test_img_u)
        else:
            return len(self.val_img_u)

    
    def __getitem__(self, idx):
        if self.task == "train":
            img_path_a = self.train_img_u[idx]
            img= self.transform(image=load_image(img_path_a))  # Implement `load_image` to load an image from a file path
            
        elif self.task == "test":
            img_path_a = self.test_img_u[idx]
            img= self.transform(image=load_image(img_path_a))
            
        else:
            img_path_a = self.val_img_u[idx]
            img= self.transform(image=load_image(img_path_a)) 
        return img["image"]
    

def plot_images_from_dataloader(dataloader, num_images=8):
    """
    Plota imagens de um dataloader.
    
    Args:
        dataloader: Um dataloader do PyTorch.
        num_images: Número de imagens a serem exibidas (padrão: 8).
    """
    # Obter um batch de imagens
    data_iter = iter(dataloader)
    images = next(data_iter)  # Pega o primeiro batch do dataloader
    
    # Certifique-se de que as imagens estão no formato correto
    images = images[:num_images]  # Seleciona o número desejado de imagens
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # De CxHxW para HxWxC
    
    # Criar o grid para plotar as imagens
    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, img in enumerate(images):
        # Normalizar os valores da imagem para [0, 1], se necessário
        img = (img - img.min()) / (img.max() - img.min())
        axes[i].imshow(img)
        axes[i].axis("off")
    
    # Preencher subplots vazios (se o número de imagens for menor que o grid)
    for j in range(len(images), len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()



