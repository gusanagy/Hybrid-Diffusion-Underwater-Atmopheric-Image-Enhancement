
import sys
import os
import wandb
# Adiciona o diretório pai ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Loss.loss import *

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
#from matplotlib import pyplot as plt
from torchvision.utils import save_image
from albumentations.pytorch import ToTensorV2
import albumentations as A
from diffusion.Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from diffusion.Model import DynamicUNet
from .Scheduler import GradualWarmupScheduler
#from tensorboardX import SummaryWriter #provavelmente irei retirar o suporte a tensorboard
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM 
from metrics.metrics import *
import numpy as np
import cv2
import colorsys
from typing import Dict, List
import PIL
from PIL import Image
import lpips
import time
import argparse
from tqdm import tqdm
import wandb
import random
import matplotlib.pyplot as plt
from utils.utils import *
from torchvision.transforms.functional import rgb_to_grayscale
from metrics.metrics import *
#from itertools import cycle


########################################
### Novas funcoes para o treinamento ###
########################################

def process_batch(input, label, device, trainer, optimizer, net_model, config, e, stage, state):
    """
    Processa um batch: move para o dispositivo, calcula perdas e realiza o backward pass.
    """
    #loss, mse_loss, perceptual_vgg, perceptual_dino, msssim, charbonnier, col_loss = None, None, None, None, None, None, None
    input, label = input.to(device), label.to(device)
    #print("input shape: ",input.shape,"label shape: ",label.shape)
    optimizer.zero_grad()
    
    [loss, mse_loss, perceptual_dino, msssim, col_loss] = trainer(input, label, stage)

    if torch.isnan(loss).any():
        print(f"Warning: NaN detected in loss at epoch {e}, stage {stage}. Skipping batch.")
    
    if state == "Train":
        # Backpropagation e atualização dos parâmetros
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), config.grad_clip)
        optimizer.step()

    
    return loss, mse_loss, perceptual_dino, msssim, col_loss

def log_metrics(wandb_, epoch_local, epoch_global, loss, mse_loss, perceptual_dino,
                msssim, col_loss, optimizer, tqdmDataLoader, num, stage, task):
    """
    Registra métricas no tqdm e no WandB, com informações claras de época e batch.
    """
    metrics = {
        "loss": None,
        "mse_loss": None,
        "Perceptual_dino": None,
        "MS_SSIM": None,
        "ang_color_loss": None
    }

    for key, value in zip(metrics.keys(), [loss, mse_loss, perceptual_dino, msssim, col_loss]):
        try:
            metrics[key] = value.mean().item()
        except Exception:
            pass  # Métrica inválida ou erro ao calcular

    # Atualiza barra de progresso com info legível
    tqdmDataLoader.set_postfix(ordered_dict={
        "ep": epoch_local,
        "ep_glob": epoch_global,
        "batch": num + 1,
        "LR": optimizer.state_dict()['param_groups'][0]["lr"],
        **{k: (f"{v:.4f}" if v is not None else "Err") for k, v in metrics.items()}
    })

    # Loga no wandb com epoch global/local e batch
    if wandb_:
        wandb.log({
            f"{task} {stage}": {
                "epoch_local": epoch_local,
                "epoch_global": epoch_global,
                "batch": num + 1,
                "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                **{k: v for k, v in metrics.items() if v is not None}
            }
        })

def train_with_dataloaders(dataloaders, device, trainer, optimizer, net_model,
                           config, epoch_local, epoch_global, num, stage, task):
    """
    Treina intercalando dataloaders, logando corretamente batch e época.
    """
    iterators = [iter(d) for d in dataloaders]
    active = [True] * len(dataloaders)

    with tqdm(total=sum(len(d) for d in dataloaders), dynamic_ncols=True) as tqdmDataLoader:
        while any(active):
            for i, iterator in enumerate(iterators):
                if not active[i]:
                    continue

                try:
                    input, label = next(iterator)
                    loss, mse_loss, perceptual_dino, msssim, col_loss = process_batch(
                        input, label, device, trainer, optimizer, net_model,
                        config, epoch_local, stage, task
                    )

                    log_metrics(config.wandb, epoch_local, epoch_global, loss, mse_loss,
                                perceptual_dino, msssim, col_loss,
                                optimizer, tqdmDataLoader, num, stage, task)

                    num += 1
                    tqdmDataLoader.update(1)

                except StopIteration:
                    active[i] = False

    return num

def val_with_dataloaders(dataloaders, device, trainer, optimizer, net_model,
                         config, epoch_local, epoch_global, num, stage):
    """
    Validação do modelo com logging de épocas e batches.
    """
    iterators = [iter(d) for d in dataloaders]
    active = [True] * len(dataloaders)

    last_loss = None

    with tqdm(total=sum(len(d) for d in dataloaders), dynamic_ncols=True) as tqdmDataLoader:
        while any(active):
            for i, iterator in enumerate(iterators):
                if not active[i]:
                    continue

                try:
                    input, label = next(iterator)
                    loss, mse_loss, perceptual_dino, msssim, col_loss = process_batch(
                        input, label, device, trainer, optimizer, net_model,
                        config, epoch_local, stage, state="val"
                    )

                    log_metrics(config.wandb, epoch_local, epoch_global, loss, mse_loss,
                                perceptual_dino, msssim, col_loss,
                                optimizer, tqdmDataLoader, num, stage, task="Val")

                    num += 1
                    tqdmDataLoader.update(1)
                    last_loss = loss.mean().item()

                except StopIteration:
                    active[i] = False

    return last_loss, num
def test_with_dataloaders(dataloaders, device, trainer, optimizer, net_model,
                          config, epoch_local, epoch_global, num, stage):
    """
    Teste final do modelo com logging consistente.
    """
    iterators = [iter(d) for d in dataloaders]
    active = [True] * len(dataloaders)

    last_loss = None

    with tqdm(total=sum(len(d) for d in dataloaders), dynamic_ncols=True) as tqdmDataLoader:
        while any(active):
            for i, iterator in enumerate(iterators):
                if not active[i]:
                    continue

                try:
                    input, label = next(iterator)
                    loss, mse_loss, perceptual_dino, msssim, col_loss = process_batch(
                        input, label, device, trainer, optimizer, net_model,
                        config, epoch_local, stage, state="test"
                    )

                    log_metrics(config.wandb, epoch_local, epoch_global, loss, mse_loss,
                                perceptual_dino, msssim, col_loss,
                                optimizer, tqdmDataLoader, num, stage, task="Test")

                    num += 1
                    tqdmDataLoader.update(1)
                    last_loss = loss.mean().item()

                except StopIteration:
                    active[i] = False

    return last_loss, num


def save_checkpoint(net_model, ckpt_savedir, e, config, stage, dataset_name):
    """
    Salva o estado do modelo em um checkpoint.
    """
    checkpoint_path = os.path.join(ckpt_savedir, f'ckpt_{e}_{stage}_{dataset_name}.pt')
    if config.DDP:
        if dist.get_rank() == 0:
            torch.save(net_model.state_dict(), checkpoint_path)
    else:
        torch.save(net_model.state_dict(), checkpoint_path)


################################
### Treinamento principal ######
################################
# Treinamento principal Train + validation => Best_Checkpoint
def train(config: Dict):
    if config.DDP:
        local_rank = int(os.getenv('LOCAL_RANK', -1))
        print('Local rank:', local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
    

    #######################################################
    #### Inicialização dos dados para a rotina de treino ###
    #######################################################
    underwater_data_train = Underwater_Dataset(config.underwater_data_name,task="train")
    atmospheric_data_train = Atmospheric_Dataset(config.atmospheric_data_name, task="train")
    underwater_data_test = Underwater_Dataset(config.underwater_data_name,task="test")
    atmospheric_data_test = Atmospheric_Dataset(config.atmospheric_data_name, task="test")

    if config.DDP:
        train_sampler_u = torch.utils.data.distributed.DistributedSampler(underwater_data_train)
        train_sampler_a = torch.utils.data.distributed.DistributedSampler(atmospheric_data_train)
        test_sampler_u = torch.utils.data.distributed.DistributedSampler(underwater_data_test)
        test_sampler_a = torch.utils.data.distributed.DistributedSampler(atmospheric_data_test)
        dataloader_u = DataLoader(underwater_data_train, batch_size=config.batch_size, sampler=train_sampler_u, 
                                  num_workers=4, drop_last=True, pin_memory=True)
        dataloader_a = DataLoader(atmospheric_data_train, batch_size=config.batch_size, sampler=train_sampler_a, 
                                  num_workers=4, drop_last=True, pin_memory=True)
        dataloader_u_test = DataLoader(underwater_data_test, batch_size=config.batch_size, sampler=test_sampler_u, 
                                  num_workers=4, drop_last=True, pin_memory=True)
        dataloader_a_test = DataLoader(atmospheric_data_test, batch_size=config.batch_size, sampler=test_sampler_a, 
                                  num_workers=4, drop_last=True, pin_memory=True)
    else:
        dataloader_u = DataLoader(underwater_data_train, batch_size=config.batch_size, num_workers=4, drop_last=True, pin_memory=True)
        dataloader_a = DataLoader(atmospheric_data_train, batch_size=config.batch_size, num_workers=4, drop_last=True, pin_memory=True)
        dataloader_u_test = DataLoader(underwater_data_test, batch_size=config.batch_size, num_workers=4, drop_last=True, pin_memory=True)
        dataloader_a_test = DataLoader(atmospheric_data_test, batch_size=config.batch_size, num_workers=4, drop_last=True, pin_memory=True)

    ###################################################
    ### Inicialização do modelo, otimizador e trainer #
    ###################################################
    
    net_model = DynamicUNet(T=config.T, ch=config.channel, ch_mult=config.channel_mult,
                            num_res_blocks=config.num_res_blocks, dropout=config.dropout)
    
    if config.pretrained_path is not None:
        ckpt = torch.load(config.pretrained_path, map_location='cpu')
        net_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})

    if config.DDP:
        net_model = DDP(net_model.cuda(), device_ids=[local_rank], output_device=local_rank)
    else:
        if len(config.device_list) > 1:
            print("Using DataParallel with devices:", config.device_list)
            device = torch.device(f"cuda:{config.device_list[0]}")
            net_model = torch.nn.DataParallel(net_model, device_ids=config.device_list).to(device)
        else:
            device=config.device_list[0]
            net_model.to(device)

    trainer = GaussianDiffusionTrainer(net_model, config.beta_1, config.beta_T, config.T).to(device)

    log_savedir = os.path.join(config.output_path, 'logs')
    os.makedirs(log_savedir, exist_ok=True)

    ckpt_savedir = os.path.join(config.output_path, 'ckpt')
    os.makedirs(ckpt_savedir, exist_ok=True)

    ######################################
    ### Definir estágios do treinamento###
    ######################################
    # stages_old = [
   
    # Aprendizado em dois passos usando gerenciamento do scheduller
    stages = [
        {"name": "Atmosferic", "lr": config.lr, "epochs": config.epochs_stage_1, "number" : int(0)},
        {"name": "Underwater", "lr": config.lr, "epochs": config.epochs_stage_2, "number" : int(1)}
    ]#é mais estável manter o mesmo otimizador e só ajustar o lr e weight_decay se for necessário.

    ################################
    #### Início do treinamento #####
    ################################

    total_epochs = 0 #o probleema esta em percorrer os datasets de teste
    num = 0 #talvez eu tenha que reestruturar o dataset e a forma como e ele e carregado 
    epoch_global = 0
    epoch_local = 0


    for stage in stages:
        print(f"Starting stage: {stage['name']} with LR: {stage['lr']} for {stage['epochs']} epochs, Identificador {stage['number']}\n")

        # Atualizar otimizador e scheduler para o estágio atual
        optimizer = torch.optim.AdamW(net_model.parameters(), lr=stage["lr"], weight_decay=1e-4)
        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=stage["epochs"], eta_min=0, last_epoch=-1)
        warmUpScheduler = GradualWarmupScheduler(
            optimizer=optimizer, multiplier=config.multiplier, 
            warm_epoch=stage["epochs"] // 10, after_scheduler=cosineScheduler)

        # Seleciona os dataloaders apropriados para treino e teste
        if "Atmosferic" in stage["name"]:
            dataloader_train = dataloader_a
            dataloader_test = dataloader_a_test
            dataset_name = config.atmospheric_data_name
        elif "Underwater" in stage["name"]:
            dataloader_train = dataloader_u
            dataloader_test = dataloader_u_test
            dataset_name = config.atmospheric_data_name + config.underwater_data_name
        else:
            raise ValueError(f"Nome de estágio inválido: {stage['name']}")
        
        for epoch_local in range(stage["epochs"]):
            epoch_global = total_epochs + epoch_local
            # Ajusta samplers por época se DDP estiver ativado
            if config.DDP and hasattr(dataloader_train, 'sampler'):
                dataloader_train.sampler.set_epoch(epoch_global)
            # Proteção contra erro se o modelo estiver no modo train
            
            num = train_with_dataloaders(
                                        dataloaders=[dataloader_train],
                                        device=device,
                                        trainer=trainer,
                                        optimizer=optimizer,
                                        net_model=net_model,
                                        config=config,
                                        epoch_local=epoch_local,
                                        epoch_global=epoch_global,
                                        num=num,
                                        stage=stage["number"],
                                        task="Train"
                                        )                               

            # Atualizar scheduler
            warmUpScheduler.step()

            # Salvar checkpoints a cada 200 épocas globais 
            if epoch_global % config.save_checkpoint == 0 or epoch_local == stage["epochs"] - 1:
                
                 # Logar métricas de teste a cada 10 épocas
                with torch.no_grad():
                    # Testar com os DataLoaders de teste
                    loss, num = val_with_dataloaders(
                        dataloaders=[dataloader_test],
                        device=device,
                        trainer=trainer,
                        optimizer=optimizer,
                        net_model=net_model,
                        config=config,
                        epoch_local=epoch_local,
                        epoch_global=epoch_global,
                        num=num,
                        stage=stage["number"]
                    )

                    if loss < best_loss:
                        best_loss = loss
                        save_checkpoint(net_model,
                                        ckpt_savedir,
                                        epoch_global,
                                        config,
                                        stage=stage["name"],
                                        dataset_name="BEST_" + stage["name"] + "_" + dataset_name)

            if config.wandb:
                wandb.alert(
                    title="🌟 Novo Melhor Modelo!",
                    text=f"Melhor MSE: {loss:.5f} na época {epoch_global} do estágio {stage['name']}",
                    level=wandb.AlertLevel.INFO
                )
              

        total_epochs += stage["epochs"]
    #save_checkpoint(net_model, ckpt_savedir, total_epochs, config, stage = "final", dataset_name=config.underwater_data_name+config.atmospheric_data_name)
    print("Training completed.")

##########################
### Teste e Inferencia ###
##########################

##copiar a funcao de treino e fazer um treino inferencia 
#incompleto
def process_batch_inference(sampler, input, label, device, net_model, config, stage):
    """
    Processa um batch para inferência: move para o dispositivo, faz a inferência e calcula as métricas.
    """
    input, label = input.to(device), label.to(device)
    
    # Colocar o modelo em modo de avaliação
    net_model.eval()
    
    # Inferência: passagem direta (sem cálculo de perdas ou backprop)
    with torch.no_grad():
        output = net_model(input)
    
    # Calcular métricas (exemplo: MSE, SSIM, etc.)
    [] = sampler(input, label, stage)

    return 

def log_metrics_inference(wandb_, loss, mse_loss, perceptual_vgg, perceptual_dino, msssim, charbonnier, col_loss, tqdmDataLoader, num, stage):
    """
    Registra as métricas durante a inferência.
    """
    metrics = {
        "loss": None,
        "mse_loss": None,
        "Perceptual_dino": None,
        "Perceptual_vgg": None,
        "MS_SSIM": None,
        "Charbonnier": None,
        "ang_color_loss": None
    }

    # Processar cada métrica
    for key, value in zip(metrics.keys(), [loss, mse_loss, perceptual_dino, perceptual_vgg, msssim, col_loss]):
        try:
            metrics[key] = value.mean().item()
        except Exception as ex:
            pass

    # Atualizar tqdmDataLoader
    tqdmDataLoader.set_postfix(ordered_dict={
        "num": num + 1,
        **{k: (v if v is not None else "Error") for k, v in metrics.items()},
    })

    # Registrar no wandb
    if wandb_:
        wandb.log({f"Inference {stage}": {**metrics, "num": num + 1}})

def inference_with_dataloaders(dataloaders, device, net_model, config, stage):
    """
    Realiza a inferência no modelo alternando entre DataLoaders.
    """
    iterators = [iter(dataloader) for dataloader in dataloaders]
    active_loaders = [True] * len(dataloaders)  # Marca quais DataLoaders ainda têm batches

    num = 0
    with tqdm(total=sum(len(dataloader) for dataloader in dataloaders), dynamic_ncols=True) as tqdmDataLoader:
        while any(active_loaders):  # Continua enquanto houver loaders ativos
            for i, iterator in enumerate(iterators):
                if not active_loaders[i]:
                    continue  # Pule se o DataLoader já foi completamente percorrido

                try:
                    input, label = next(iterator)  # Obter o próximo batch
                    
                    # Processar batch de inferência
                    loss, mse_loss, perceptual_vgg, perceptual_dino, msssim, charbonnier, col_loss, output = process_batch_inference(
                        input, label, device, net_model, config, stage
                    )

                    # Logar métricas
                    
                    # Salvar as imagens
                    save_images(output, input, label, config, stage, num)

                    num += 1
                    tqdmDataLoader.update(1)

                except StopIteration:
                    # Marcar o DataLoader como concluído
                    active_loaders[i] = False

    return num


    
#Precisa de ajustes para funcionar como a funcao de treino
def test(config: Dict,epoch):
   
    #######################################################
    #### Inicialização dos dados para a rotina de treino ###
    #######################################################
    underwater_data = Underwater_Dataset(config.underwater_data_name,task="test")
    atmospheric_data = Atmospheric_Dataset(config.atmospheric_data_name,task="test")

    dataloader_u = DataLoader(underwater_data, batch_size=config.batch_size, num_workers=4, drop_last=True, pin_memory=True)
    dataloader_a = DataLoader(atmospheric_data, batch_size=config.batch_size, num_workers=4, drop_last=True, pin_memory=True)

    device = config.device_list[0]  # Usando o primeiro dispositivo disponível
    
    #####################################################
    ### Inicialização do modelo, otimizador e trainer ###
    #####################################################

    model = DynamicUNet(T=config.T, ch=config.channel, ch_mult=config.channel_mult,
                 num_res_blocks=config.num_res_blocks, dropout=0.)
    #Mudar um pouco aqui para carregar o checkpoint do dataset escolhido
    ckpt = torch.load(config.pretrained_path,map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    print("model load weight done.")


    save_dir_u="output/result/"+ config.pretrained_path.split('/')[-1] +'/'+config.underwater_data_name+"/"
    save_dir_a="output/result/"+ config.pretrained_path.split('/')[-1] +'/'+config.atmospheric_data_name+"/"
    print(save_dir_a, save_dir_u)
    if not os.path.exists(save_dir_u):
        os.makedirs(save_dir_u)
    if not os.path.exists(save_dir_a):
        os.makedirs(save_dir_a)

    print(f"Save dir underwater for combination {config.underwater_data_name+config.atmospheric_data_name}: {save_dir_u}")
    print(f"Save dir atmospheric for combination {config.underwater_data_name+config.atmospheric_data_name}: {save_dir_a}")

    save_txt_name_u =save_dir_u + 'res.txt'
    save_txt_name_a =save_dir_a + 'res.txt'
    f = open(save_txt_name_u, 'w+');    f.close()
    f = open(save_txt_name_a, 'w+');    f.close()
    fid_score = FID(device=device)
    image_num = 0
    psnr_list = []
    ssim_list = []
    uciqe_list = []
    uiconm_list = []
    uism_list = []
    uicm_list =[]
    uiqm_list =[]
    fid_list = []
    wout = []

 
    model.eval()
    sampler = GaussianDiffusionSampler(
        model, config.beta_1, config.beta_T, config.T).to(device)
    print("model load weight done.")
    print(f"Avaliando Modelo {config.pretrained_path.split('/')[-1]} subaquatico {config.underwater_data_name}\n")
    with torch.no_grad():
        with tqdm( dataloader_u, dynamic_ncols=True) as tqdmDataLoader:
                image_num = 0
                for input_image, gt_image, name in tqdmDataLoader:
                    #print('Image:',name)

                    gt_image = gt_image.to(device)
                    input_image = input_image.to(device)
                        
                    time_start = time.time()
                    sampledImgs = sampler(input_image,ddim=True,
                                          unconditional_guidance_scale=1,ddim_step=config.ddim_step)
                    time_end=time.time()
                    print('time cost:', time_end - time_start, "\n")

                    sampledImgs=(sampledImgs+1)/2
                    
                    fid_score = fid_score.compute_fid(sampledImgs,gt_image)
                    #print(sampledImgs.shape, res_Imgs.shape, gt_img.shape, input_image.shape)
                    for i in range(sampledImgs.shape[0]):
                        
                        res_Imgs = np.clip(sampledImgs[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                        gt_img = np.clip(gt_image[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                        input_image = np.clip(input_image[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                                              
                        psnr = PSNR(res_Imgs, gt_img,data_range=255)
                        uiqm0,uciqe0,uism,uicm,uiconm = nmetrics(res_Imgs)
                        #uciqe1 = uciqe(nargin=1,loc=res_Imgs)

                        ssim = SSIM(res_Imgs, gt_img, channel_axis=2,data_range=255,multichannel=True)
                        #uciqe2 = uciqe(nargin=1,loc=res_Imgs)#usarei este
                        uiqm1 = getUIQM(res_Imgs)
                        

                        uciqe_list.append(uciqe0)  
                        uiqm_list.append(uiqm1)
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        fid_list.append(fid_score)
                        uism_list.append(uism)
                        uicm_list.append(uicm)
                        uiconm_list.append(uiconm)
                        cv2.imwrite(save_dir_u+name[i],res_Imgs)
                        
                        # print(f"""
                        #       uiqm0 : {uiqm0},  
                        #       uiqm1: {uiqm1}, usar
                        #       uciqe0: {uciqe0},

                        #       uism: {uism},
                        #       uicm: {uicm},
                        #       uiconm: {uiconm},

                        #       ssim: {ssim}, 
                        #       psnr: {psnr}, 
                        #       fid: {fid_score}, 
                        # """)

                #AVERAGE SSIM PSNR UICM UCIQE
                avg_psnr = sum(psnr_list) / len(psnr_list)
                avg_ssim = sum(ssim_list) / len(ssim_list)
                avg_uiqm = sum(uiqm_list) / len(uiqm_list)
                avg_uciqe = sum(uciqe_list) / len(uciqe_list)       
                avg_uism = sum(uism_list) / len(uism_list)    
                avg_uicm = sum(uicm_list) / len(uicm_list)
                avg_fid = sum(fid_list) / len(fid_list)
                avg_uiconm = sum(uiconm_list) / len(uiconm_list)
                        
                f = open(save_txt_name_u, 'w+')

                f.write('\npsnr_orgin_avg:')
                f.write(str(avg_psnr))
                f.write('\nssim_orgin_avg:')
                f.write(str(avg_ssim))
                f.write('\nfid_orgin_avg:')
                f.write(str(avg_fid))
                f.write('\nuiqm_orgin_avg:')
                f.write(str(avg_uiqm))
                f.write('\nuciqe_orgin_avg:')
                f.write(str(avg_uciqe))
                f.write('\nuism_orgin_avg:')
                f.write(str(avg_uism))
                f.write('\nuicm_orgin_avg:')
                f.write(str(avg_uicm))
                f.write('\nuiconm_orgin_avg:')
                f.write(str(avg_uiconm))


                f.close()
        #reinicieando listas de dados
        psnr_list = []
        ssim_list = []
        uciqe_list = []
        uiconm_list = []
        uism_list = []
        uicm_list =[]
        uiqm_list =[]
        fid_list = []
        print(f"Avaliando Modelo {config.pretrained_path.split('/')[-1]} subaquatico {config.underwater_data_name}\n")
        with tqdm( dataloader_a, dynamic_ncols=True) as tqdmDataLoader:
                image_num = 0
                for input_image, gt_image, name in tqdmDataLoader:
                    #print('Image:',name)

                    gt_image = gt_image.to(device)
                    input_image = input_image.to(device)
                        
                    time_start = time.time()
                    sampledImgs = sampler(input_image,ddim=True,
                                          unconditional_guidance_scale=1,ddim_step=config.ddim_step)
                    time_end=time.time()
                    print('time cost:', time_end - time_start, "\n")

                    sampledImgs=(sampledImgs+1)/2
                    
                    fid_score = fid_score.compute_fid(sampledImgs,gt_image)
                    #print(sampledImgs.shape, res_Imgs.shape, gt_img.shape, input_image.shape)
                    for i in range(sampledImgs.shape[0]):
                        
                        res_Imgs = np.clip(sampledImgs[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                        gt_img = np.clip(gt_image[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                        input_image = np.clip(input_image[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                                              
                        psnr = PSNR(res_Imgs, gt_img,data_range=255)
                        uiqm0,uciqe0,uism,uicm,uiconm = nmetrics(res_Imgs)
                        #uciqe1 = uciqe(nargin=1,loc=res_Imgs)

                        ssim = SSIM(res_Imgs, gt_img, channel_axis=2,data_range=255,multichannel=True)
                        #uciqe2 = uciqe(nargin=1,loc=res_Imgs)#usarei este
                        uiqm1 = getUIQM(res_Imgs)
                        

                        uciqe_list.append(uciqe0)  
                        uiqm_list.append(uiqm1)
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        fid_list.append(fid_score)
                        uism_list.append(uism)
                        uicm_list.append(uicm)
                        uiconm_list.append(uiconm)
                        cv2.imwrite(save_dir_a+name[i],res_Imgs)
                        
                        # print(f"""
                        #       uiqm0 : {uiqm0},  
                        #       uiqm1: {uiqm1}, usar
                        #       uciqe0: {uciqe0},

                        #       uism: {uism},
                        #       uicm: {uicm},
                        #       uiconm: {uiconm},

                        #       ssim: {ssim}, 
                        #       psnr: {psnr}, 
                        #       fid: {fid_score}, 
                        # """)

                #AVERAGE SSIM PSNR UICM UCIQE
                avg_psnr = sum(psnr_list) / len(psnr_list)
                avg_ssim = sum(ssim_list) / len(ssim_list)
                avg_uiqm = sum(uiqm_list) / len(uiqm_list)
                avg_uciqe = sum(uciqe_list) / len(uciqe_list)       
                avg_uism = sum(uism_list) / len(uism_list)    
                avg_uicm = sum(uicm_list) / len(uicm_list)
                avg_fid = sum(fid_list) / len(fid_list)
                avg_uiconm = sum(uiconm_list) / len(uiconm_list)
                        
                f = open(save_txt_name_a, 'w+')

                f.write('\npsnr_orgin_avg:')
                f.write(str(avg_psnr))
                f.write('\nssim_orgin_avg:')
                f.write(str(avg_ssim))
                f.write('\nfid_orgin_avg:')
                f.write(str(avg_fid))
                f.write('\nuiqm_orgin_avg:')
                f.write(str(avg_uiqm))
                f.write('\nuciqe_orgin_avg:')
                f.write(str(avg_uciqe))
                f.write('\nuism_orgin_avg:')
                f.write(str(avg_uism))
                f.write('\nuicm_orgin_avg:')
                f.write(str(avg_uicm))
                f.write('\nuiconm_orgin_avg:')
                f.write(str(avg_uiconm))


                f.close()
    print("Test completed.")

def inference(config: Dict,epoch):
   
    #######################################################
    #### Inicialização dos dados para a rotina de treino ###
    #######################################################
    underwater_data = Underwater_Dataset(config.underwater_data_name,task="val")
    atmospheric_data = Atmospheric_Dataset(config.atmospheric_data_name,task="val")

    dataloader_u = DataLoader(underwater_data, batch_size=config.batch_size, num_workers=4, drop_last=True, pin_memory=True)
    dataloader_a = DataLoader(atmospheric_data, batch_size=config.batch_size, num_workers=4, drop_last=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Usando o primeiro dispositivo disponível
    #device = "cpu"
    #####################################################
    ### Inicialização do modelo, otimizador e trainer ###
    #####################################################


    model = DynamicUNet(T=config.T, ch=config.channel, ch_mult=config.channel_mult,
                 num_res_blocks=config.num_res_blocks, dropout=0.)
    #Mudar um pouco aqui para carregar o checkpoint do dataset escolhido
    ckpt = torch.load(config.pretrained_path,map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    print("model load weight done.")


    save_dir_u="output/result/"+ config.pretrained_path.split('/')[-1] +'/'+config.underwater_data_name+"/"
    save_dir_a="output/result/"+ config.pretrained_path.split('/')[-1] +'/'+config.atmospheric_data_name+"/"
    print(save_dir_a, save_dir_u)
    if not os.path.exists(save_dir_u):
        os.makedirs(save_dir_u)
    if not os.path.exists(save_dir_a):
        os.makedirs(save_dir_a)

    print(f"Save dir underwater for combination {config.underwater_data_name+config.atmospheric_data_name}: {save_dir_u}")
    print(f"Save dir atmospheric for combination {config.underwater_data_name+config.atmospheric_data_name}: {save_dir_a}")

    save_txt_name_u =save_dir_u + 'res.txt'
    save_txt_name_a =save_dir_a + 'res.txt'
    f = open(save_txt_name_u, 'w+');    f.close()
    f = open(save_txt_name_a, 'w+');    f.close()
    fid_score = FID(device=device)
    image_num = 0
    psnr_list = []
    ssim_list = []
    uciqe_list = []
    uiconm_list = []
    uism_list = []
    uicm_list =[]
    uiqm_list =[]
    fid_list = []
    wout = []

 
    model.eval()
    sampler = GaussianDiffusionSampler(
        model, config.beta_1, config.beta_T, config.T).to(device)
    print("model load weight done.")
    print(f"Avaliando Modelo {config.pretrained_path.split('/')[-1]} subaquatico {config.underwater_data_name}\n")
    with torch.no_grad():
        with tqdm( dataloader_u, dynamic_ncols=True) as tqdmDataLoader:
                image_num = 0
                for input_image, gt_image, name in tqdmDataLoader:
                    #print('Image:',name)

                    gt_image = gt_image.to(device)
                    input_image = input_image.to(device)
                        
                    time_start = time.time()
                    sampledImgs = sampler(input_image,ddim=True,
                                          unconditional_guidance_scale=1,ddim_step=config.ddim_step)
                    time_end=time.time()
                    #print('time cost:', time_end - time_start, "\n")

                    sampledImgs=(sampledImgs+1)/2
                    
                    fid = fid_score.compute_fid(sampledImgs,gt_image)
                    #print(sampledImgs.shape, res_Imgs.shape, gt_img.shape, input_image.shape)
                    for i in range(sampledImgs.shape[0]):
                        
                        res_Imgs = np.clip(sampledImgs[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                        gt_img = np.clip(gt_image[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                        #input_image = np.clip(input_image[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                                              
                        psnr = PSNR(res_Imgs, gt_img,data_range=255)
                        uiqm0,uciqe0,uism,uicm,uiconm = nmetrics(res_Imgs)
                        #uciqe1 = uciqe(nargin=1,loc=res_Imgs)

                        ssim = SSIM(res_Imgs, gt_img, channel_axis=2,data_range=255,multichannel=True)
                        #uciqe2 = uciqe(nargin=1,loc=res_Imgs)#usarei este
                        uiqm1 = getUIQM(res_Imgs)
                        

                        uciqe_list.append(uciqe0)  
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        fid_list.append(fid)
                        uism_list.append(uism)
                        uicm_list.append(uicm)
                        uiconm_list.append(uiconm)
                        cv2.imwrite(save_dir_u+name[i],res_Imgs)
                        
                        # print(f"""
                        #       uiqm0 : {uiqm0},  
                        #       uiqm1: {uiqm1}, usar
                        #       uciqe0: {uciqe0},

                        #       uism: {uism},
                        #       uicm: {uicm},
                        #       uiconm: {uiconm},

                        #       ssim: {ssim}, 
                        #       psnr: {psnr}, 
                        #       fid: {fid_score}, 
                        # """)

                #AVERAGE SSIM PSNR UICM UCIQE
                print(len(psnr_list),len(ssim_list),len(uiqm_list),len(uciqe_list),len(uism_list),len(uicm_list),len(uiconm_list),len(fid_list))
                avg_psnr = (sum(psnr_list) + 1 )/ (len(psnr_list) +1)
                avg_ssim = (sum(ssim_list) +1)/ (len(ssim_list) +1 )
                avg_uiqm = (sum(uiqm_list) +1)/ (len(uiqm_list) +1 )
                avg_uciqe = (sum(uciqe_list) +1)/ (len(uciqe_list) +1 )      
                avg_uism = (sum(uism_list) +1)/ (len(uism_list) +1 )
                avg_uicm = (sum(uicm_list) +1 )/ (len(uicm_list) +1 )
                avg_fid = (sum(fid_list) +1 )/ (len(fid_list) +1 )
                avg_uiconm = (sum(uiconm_list) +1 )/ (len(uiconm_list) +1 )
                        
                f = open(save_txt_name_u, 'w+')

                f.write('\npsnr_orgin_avg:')
                f.write(str(avg_psnr))
                f.write('\nssim_orgin_avg:')
                f.write(str(avg_ssim))
                f.write('\nfid_orgin_avg:')
                f.write(str(avg_fid))
                f.write('\nuiqm_orgin_avg:')
                f.write(str(avg_uiqm))
                f.write('\nuciqe_orgin_avg:')
                f.write(str(avg_uciqe))
                f.write('\nuism_orgin_avg:')
                f.write(str(avg_uism))
                f.write('\nuicm_orgin_avg:')
                f.write(str(avg_uicm))
                f.write('\nuiconm_orgin_avg:')
                f.write(str(avg_uiconm))


                f.close()
        #reinicieando listas de dados
        psnr_list = []
        ssim_list = []
        uciqe_list = []
        uiconm_list = []
        uism_list = []
        uicm_list =[]
        uiqm_list =[]
        fid_list = []
        print(f"Avaliando Modelo {config.pretrained_path.split('/')[-1]} atmosferico {config.atmospheric_data_name}\n")
        with tqdm( dataloader_a, dynamic_ncols=True) as tqdmDataLoader:
                image_num = 0
                for input_image, gt_image, name in tqdmDataLoader:
                    #print('Image:',name)

                    gt_image = gt_image.to(device)
                    input_image = input_image.to(device)
                        
                    time_start = time.time()
                    sampledImgs = sampler(input_image,ddim=True,
                                          unconditional_guidance_scale=1,ddim_step=config.ddim_step)
                    time_end=time.time()
                    #print('time cost:', time_end - time_start, "\n")

                    sampledImgs=(sampledImgs+1)/2
                    
                    fid = fid_score.compute_fid(sampledImgs,gt_image)
                    #print(sampledImgs.shape, res_Imgs.shape, gt_img.shape, input_image.shape)
                    for i in range(sampledImgs.shape[0]):
                        
                        res_Imgs = np.clip(sampledImgs[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                        gt_img = np.clip(gt_image[i].detach().cpu().numpy().transpose(1, 2, 0),0,1)*255#[:,:,::-1]
                                              
                        psnr = PSNR(res_Imgs, gt_img,data_range=255)
                        uiqm0,uciqe0,uism,uicm,uiconm = nmetrics(res_Imgs)
                        #uciqe1 = uciqe(nargin=1,loc=res_Imgs)

                        ssim = SSIM(res_Imgs, gt_img, channel_axis=2,data_range=255,multichannel=True)
                        #uciqe2 = uciqe(nargin=1,loc=res_Imgs)#usarei este
                        uiqm1 = getUIQM(res_Imgs)
                        

                        uciqe_list.append(uciqe0)  
                        uiqm_list.append(uiqm1)
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        fid_list.append(fid)
                        uism_list.append(uism)
                        uicm_list.append(uicm)
                        uiconm_list.append(uiconm)
                        cv2.imwrite(save_dir_a+name[i],res_Imgs)
                        
                        # print(f"""
                        #       uiqm0 : {uiqm0},  
                        #       uiqm1: {uiqm1}, usar
                        #       uciqe0: {uciqe0},

                        #       uism: {uism},
                        #       uicm: {uicm},
                        #       uiconm: {uiconm},

                        #       ssim: {ssim}, 
                        #       psnr: {psnr}, 
                        #       fid: {fid_score}, 
                        # """)
                        
                    
                #AVERAGE SSIM PSNR UICM UCIQE
                avg_psnr = (sum(psnr_list) + 1 )/ (len(psnr_list) +1)
                avg_ssim = (sum(ssim_list) +1)/ (len(ssim_list) +1 )
                avg_uiqm = (sum(uiqm_list) +1)/ (len(uiqm_list) +1 )
                avg_uciqe = (sum(uciqe_list) +1)/ (len(uciqe_list) +1 )      
                avg_uism = (sum(uism_list) +1)/ (len(uism_list) +1 )
                avg_uicm = (sum(uicm_list) +1 )/ (len(uicm_list) +1 )
                avg_fid = (sum(fid_list) +1 )/ (len(fid_list) +1 )
                avg_uiconm = (sum(uiconm_list) +1 )/ (len(uiconm_list) +1 )
                        
                f = open(save_txt_name_a, 'w+')

                f.write('\npsnr_orgin_avg:')
                f.write(str(avg_psnr))
                f.write('\nssim_orgin_avg:')
                f.write(str(avg_ssim))
                f.write('\nfid_orgin_avg:')
                f.write(str(avg_fid))
                f.write('\nuiqm_orgin_avg:')
                f.write(str(avg_uiqm))
                f.write('\nuciqe_orgin_avg:')
                f.write(str(avg_uciqe))
                f.write('\nuism_orgin_avg:')
                f.write(str(avg_uism))
                f.write('\nuicm_orgin_avg:')
                f.write(str(avg_uicm))
                f.write('\nuiconm_orgin_avg:')
                f.write(str(avg_uiconm))


                f.close()
    print("Test completed.")

if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    modelConfig = {
  
        "DDP": False,
        "state": "eval", # or eval
        "epoch": 601,#10001,
        "batch_size":16 ,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda", #MODIFIQUEI
        "device_list": [0],
        #"device_list": [3,2,1,0],
        
        "ddim":True,
        "unconditional_guidance_scale":1,
        "ddim_step":100
    }


    parser.add_argument('--dataset_path', type=str, default="./data/UDWdata/")
    parser.add_argument('--dataset', type=str, default="all") # RUIE, UIEB, SUIM
    parser.add_argument('--state', type=str, default="train")  #or eval
    parser.add_argument('--pretrained_path', type=str, default=None)  #or eval ajustar pastas para salvar os conteudos
    parser.add_argument('--output_path', type=str, default="./output/")  #or eval

    config = parser.parse_args()
    
    # wandb.init(
    #         project="CLEDiffusion",
    #         config=vars(config),
    #         name="Treino Diffusao sem mascaras",
    #         tags=["Train","No mask"],
    #         group="diffusion_train",
    #         job_type="train",

        # )
    
    for key, value in modelConfig.items():
        setattr(config, key, value)
    print(config)
    test(config,1000)
    # wandb.finish()
    #Test_for_one(modelConfig,epoch=14000)
