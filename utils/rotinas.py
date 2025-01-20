
import sys
import os

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
from diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from diffusion.Model import UNet
from .Scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter #provavelmente irei retirar o suporte a tensorboard
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM 
from metrics.metrics import *
import numpy as np
import random
import cv2
import colorsys
import os
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


def train(config: Dict):
    if config.DDP==True:
        local_rank = int(os.getenv('LOCAL_RANK', -1))
        print('locak rank:',local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
    
    ###load the data
    datapath_train = load_image_paths(config.dataset_path,config.dataset)
    dataload_train=load_data(datapath_train, datapath_train)

    ###Modificar aqui a forma como sao carregados os parametros
    if config.DDP == True:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataload_train)
        dataloader= DataLoader(dataload_train, batch_size=config.batch_size,sampler=train_sampler)
    else:
        dataloader = DataLoader(dataload_train, batch_size=config.batch_size, shuffle=True, num_workers=4,
                                drop_last=True, pin_memory=True)
    ###carrega o modelo com as configuracoes indicadas 
    net_model = UNet(T=config.T, ch=config.channel, ch_mult=config.channel_mult, attn=config.attn,
                     num_res_blocks=config.num_res_blocks, dropout=config.dropout)

    if config.pretrained_path is not None:
        ckpt = torch.load(os.path.join(
                config.pretrained_path), map_location='cpu')
        net_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})


    if config.DDP == True:
        net_model = DDP(net_model.cuda(), device_ids=[local_rank], output_device=local_rank,)
    else:
        net_model=torch.nn.DataParallel(net_model,device_ids=config.device_list)
        device=config.device_list[0]
        net_model.to(device)

    ##Set modeltools
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=config.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=config.multiplier, warm_epoch=config.epoch // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, config.beta_1, config.beta_T, config.T,perceptual='alex',).to(device)


    log_savedir=config.output_path+'/logs/'
    if not os.path.exists(log_savedir):
        os.makedirs(log_savedir)
    #writer = SummaryWriter(log_dir=log_savedir)#sumario de escrita 

    ckpt_savedir=config.output_path+'/ckpt/'
    if not os.path.exists(ckpt_savedir):
        os.makedirs(ckpt_savedir)
    #save_txt= config.output_path + 'res.txt'

    #### Start training routine
    ###Modificar rotina de treino e forma como tqdm funciona // Inserir teste das novas metricas no treinamento
    num=0
    for e in range(config.epoch):
        if config.DDP == True:
           dataloader.sampler.set_epoch(e)

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:##nao ajustar loacal do tqdm para cima//usa a estrtura do posfix
            for data_low, data_high, data_color, data_blur in tqdmDataLoader:
                data_high = data_high.to(device)
                data_low = data_low.to(device)
                data_color=data_color.to(device)
                data_blur=data_blur.to(device)#printar processamento psnr e concatenacoes
                snr_map = getSnrMap(data_low, data_blur)
                data_concate=torch.cat([data_color, snr_map], dim=1)
                optimizer.zero_grad()

                [loss, mse_loss, col_loss, exp_loss, ssim_loss, perceptual_loss] = trainer(data_high, data_low,data_concate,e)
                #[loss, mse_loss, col_loss,exp_loss,ssim_loss,vgg_loss] = trainer(data_high, data_low,data_concate,e)
                ###calcula a media das funcoes de perda apos os passos do sampler
                loss = loss.mean()
                mse_loss = mse_loss.mean()
                ssim_loss= ssim_loss.mean()
                perceptual_loss = perceptual_loss.mean()
                

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), config.grad_clip)
                optimizer.step()
                ###Entender esta linha
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "mse_loss":mse_loss.item(),
                    "Brithness_loss":exp_loss.item(),
                    "col_loss":col_loss.item(),
                    'ssim_loss':ssim_loss.item(),
                    'perceptual_loss':perceptual_loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                    "num":num+1
                })
                loss_num=loss.item()
                mse_num=mse_loss.item()
                exp_num=exp_loss.item()
                col_num=col_loss.item()
                ssim_num = ssim_loss.item()
                perceptual_num=perceptual_loss.item()
                #L1_num = L1loss.item()
                # writer.add_scalars('loss', {"loss_total":loss_num,
                #                              "mse_loss":mse_num,
                #                              "exp_loss":exp_num,
                #                             'ssim_loss':ssim_num,
                #                              "col_loss":col_num,
                #                             "vgg_loss":vgg_num,
                #                               }, num)
                #Wandb Logs 
                wandb.log({"Train":{
                    "epoch": e,
                    "Loss: ": loss_num,
                    "MSE Loss":mse_num,
                    "Brithness_loss":exp_num,
                    "COL Loss":col_num,
                    'SSIM Loss':ssim_num,
                    'perceptual Loss':perceptual_num,
                    }})
                num+=1
                #Adicionar uma flag do wandb para acompanhar a loss// adaptar o summary writer do tensor board

        warmUpScheduler.step()
      
        if e % 200 == 0:
            if config.DDP == True:
                if dist.get_rank() == 0:
                    torch.save(net_model.state_dict(), os.path.join(
                        ckpt_savedir, 'ckpt_' + str(e) + "_.pt"))
            elif config.DDP == False:
                torch.save(net_model.state_dict(), os.path.join(
                    ckpt_savedir, 'ckpt_' + str(e) + "_.pt"))
            ##TEST FUNCTION
            

        # if e % 200==0 and  e > 10:
        #     Test(config,e)
            #avg_psnr,avg_ssim=Test(config,e)
            #write_data = 'epoch: {}  psnr: {:.4f} ssim: {:.4f}\n'.format(e, avg_psnr,avg_ssim)
            #f = open(save_txt, 'a+')
            #f.write(write_data)
            #f.close()

def Test(config: Dict,epoch):

    ###load the data
    datapath_test = load_image_paths(config.dataset_path,config.dataset,task="val",split=False)
    print(len(datapath_test))
    # load model and evaluate
    device = config.device_list[0]
    # test_low_path=config.dataset_path+r'*.png'    
    # test_high_path=config.dataset_path+r'*.png' 

    # datapath_test_low = glob.glob( test_low_path)
    # datapath_test_high = glob.glob(test_high_path)

    dataload_test = load_data_test(datapath_test,datapath_test)
    dataloader = DataLoader(dataload_test, batch_size=1, num_workers=4,
                            drop_last=True, pin_memory=True)


    model = UNet(T=config.T, ch=config.channel, ch_mult=config.channel_mult,
                 attn=config.attn,
                 num_res_blocks=config.num_res_blocks, dropout=0.)
    #Mudar um pouco aqui para carregar o checkpoint do dataset escolhido
    ckpt_path=config.output_path+'ckpt/'+ config.dataset +'/ckpt_'+str(epoch)+'_.pt'
    ckpt = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    print("model load weight done.")
    save_dir=config.output_path+'result/'+ config.dataset +'/epoch/'+str(epoch)+'/'
    save_concate=config.output_path+'result/'+ config.dataset +'/epoch/'+str(epoch)+'concate'+'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_concate):
        os.makedirs(save_concate)

    print(f"savedir: {save_dir}, ckpt_path: {ckpt_path}")
    save_txt_name =save_dir + 'res.txt'
    f = open(save_txt_name, 'w+')
    f.close()
        
    image_num = 0
    psnr_list = []
    ssim_list = []
    #lpips_list=[]
    uciqe_list = []
    uiqm_list =[]
    wout = []

 
    model.eval()
    sampler = GaussianDiffusionSampler(
        model, config.beta_1, config.beta_T, config.T).to(device)
    #loss_fn_vgg=lpips.LPIPS(net='vgg')
     
    with torch.no_grad():
        with tqdm( dataloader, dynamic_ncols=True) as tqdmDataLoader:
                image_num = 0
                for data_low, data_high, data_color,data_blur,filename in tqdmDataLoader:
                    name=filename[0].split('/')[-1]
                    print('Image:',name)
                    gt_image = data_high.to(device)
                    lowlight_image = data_low.to(device)
                    data_color = data_color.to(device)
                    data_blur=data_blur.to(device)
                    snr_map = getSnrMap(lowlight_image, data_blur)
                    data_concate=torch.cat([data_color, snr_map], dim=1)

                    #for i in range(-10, 10,1): 
                        # light_high = torch.ones([1]) * i*0.1
                        # light_high = light_high.to(device)
                        
                    brightness_level=gt_image.mean([1, 2, 3]) # b*1
                    time_start = time.time()
                    sampledImgs = sampler(lowlight_image, data_concate,brightness_level,ddim=True,
                                          unconditional_guidance_scale=1,ddim_step=config.ddim_step)
                    time_end=time.time()
                    print('time cost:', time_end - time_start)

                    sampledImgs=(sampledImgs+1)/2
                    gt_image=(gt_image+1)/2
                    lowlight_image=(lowlight_image+1)/2
                    res_Imgs=np.clip(sampledImgs.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1] 
                    gt_img=np.clip(gt_image.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1]
                    low_img=np.clip(lowlight_image.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1]
                    
                    
                    # Compute METRICS
                    ## compute psnr
                    psnr = PSNR(res_Imgs, gt_img)
                    #ssim = SSIM(res_Imgs, gt_img, channel_axis=2,data_range=255)
                    res_gray = rgb2gray(res_Imgs)
                    gt_gray = rgb2gray(gt_img)

                    ssim_score = SSIM(res_gray, gt_gray, multichannel=True,data_range=1)\
                    
                    psnr_list.append(psnr)
                    ssim_list.append(ssim_score)
                    
                    
                    #send wandb
                    output = np.concatenate([low_img, gt_img, res_Imgs], axis=1)*255
                    image = wandb.Image(output, caption="Low image, High Image, Enhanced Image")
                    #wout.append(image)
                    if len(wout) <= 5:
                        wout.append(image)

                    # show result
                    # output = np.concatenate([low_img, gt_img, res_Imgs, res_trick], axis=1) / 255
                    # plt.axis('off')
                    # plt.imshow(output)
                    # plt.show()
                    save_path = save_concate + name
                    cv2.imwrite(save_path, output)

                    save_path =save_dir + name
                    cv2.imwrite(save_path, res_Imgs*255)
                 
    #Metrics
    #UIQM e UCIQE
    print("Calculationg Metrics\n")
    a = list_images(save_dir)
    print(f"calculando {len(a)} amostras")
    
    for path in a:
        res_Imgs = cv2.imread(path)
        uiqm,_= nmetrics(res_Imgs)
        uciqe_ = uciqe(nargin=1,loc=res_Imgs)
        print(f"uiqm: {uiqm}, uciqe: {uciqe_}")
        uiqm_list.append(uiqm)
        uciqe_list.append(uciqe_)
    #AVERAGE SSIM PSNR UICM UCIQE
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_uiqm = sum(uiqm_list) / len(uiqm_list)
    avg_uciqe = sum(uciqe_list) / len(uciqe_list)                 

    f = open(save_txt_name, 'w+')
              
    """ f.write('\nuiqm_list :')
    f.write(str(uiqm_list))
    f.write('\nuciqe_list :')
    f.write(str(uciqe_list))
    f.write('\nuism_list :') """

    f.write('\npsnr_orgin_avg:')
    f.write(str(avg_psnr))
    f.write('\nssim_orgin_avg:')
    f.write(str(avg_ssim))
    f.write('\nuiqm_orgin_avg:')
    f.write(str(avg_uiqm))
    f.write('\nuciqe_orgin_avg:')
    f.write(str(avg_uciqe))

    f.close()

    
    # Wandb logs 
    wandb.log({"Test "+config.dataset:{
                     "Average PSNR": avg_psnr,
                     "Average SSIM": avg_ssim,
                     "Average UIQM": avg_uiqm,
                     "Average UCIQE": avg_uciqe,
                     "Test from epoch": epoch,
                     "Image ":wout
                     }})
    print(f"""
            Test From epoch {epoch} DONE 
            """)
                #return avg_psnr,avg_ssim
    #plot_images(wout)

def Inference(config: Dict,epoch):

    ###load the data
    #datapath_test = load_image_paths(dataset_path=config.dataset_path,dataset=config.dataset,split=False,task="val")[:1]
    #print(datapath_test)
    datapath_test = [config.inference_image]
    # load model and evaluate
    device = config.device_list[0]
    
    dataload_test = load_data_test(datapath_test,datapath_test)
    dataloader = DataLoader(dataload_test, batch_size=1, num_workers=4,
                            drop_last=True, pin_memory=True)

    model = UNet(T=config.T, ch=config.channel, ch_mult=config.channel_mult,
                 attn=config.attn,
                 num_res_blocks=config.num_res_blocks, dropout=0.)
    #Mudar um pouco aqui para carregar o checkpoint do dataset escolhido
    #ckpt_path=config.output_path+'ckpt/'+ config.dataset +'/ckpt_'+str(epoch)+'_.pt'
    ckpt = torch.load(config.pretrained_path ,map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    print("model load weight done.")
    save_dir=config.output_path+'result/'+ config.dataset+'/ctrl' +'/epoch/'+str(epoch)+'inf/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_txt_name =save_dir + 'res.txt'
    f = open(save_txt_name, 'w+')
    f.close()

    image_num = 0
    imags = []

    model.eval()
    sampler = GaussianDiffusionSampler(
        model, config.beta_1, config.beta_T, config.T).to(device)
    #loss_fn_vgg=lpips.LPIPS(net='vgg')
    with torch.no_grad():
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                image_num = 0
                for data_low, data_high, data_color,data_blur,filename in tqdmDataLoader:
                    name=filename[0].split('/')[-1]
                    print('Image:',name)
                    gt_image = data_high.to(device)
                    lowlight_image = data_low.to(device)
                    data_color = data_color.to(device)
                    data_blur=data_blur.to(device)
                    snr_map = getSnrMap(lowlight_image, data_blur)
                    data_concate=torch.cat([data_color, snr_map], dim=1)

                    #for i in range(-10, 10,1): 
                        # light_high = torch.ones([1]) * i*0.1
                        # light_high = light_high.to(device)
                        
                    brightness_level=gt_image.mean([1, 2, 3]) # b*1
                    #time_start = time.time()
                    sampledImgs = sampler(lowlight_image, data_concate,brightness_level,ddim=True,
                                          unconditional_guidance_scale=1,ddim_step=config.ddim_step)
                    #time_end=time.time()
                    #print('time cost:', time_end - time_start)

                    
                    
                    # for i in range(-3, 3): 
                    #     brightness_level = torch.ones([1]) * i
                    #     brightness_level = brightness_level.to(device)
                        
                    #     #time_start = time.time()
                    #     sampledImgs = sampler(lowlight_image, data_concate,brightness_level,ddim=True,
                    #                         unconditional_guidance_scale=1,ddim_step=config.ddim_step)
                    #     #time_end=time.time()
                    #     #print('time cost:', time_end - time_start)

                    sampledImgs=(sampledImgs+1)/2
                    gt_image=(gt_image+1)/2
                    lowlight_image=(lowlight_image+1)/2
                    res_Imgs=np.clip(sampledImgs.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1] 
                    #gt_img=np.clip(gt_image.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1]
                    #low_img=np.clip(lowlight_image.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1]
                    

                    #wandb.log({"Image Inference": [wandb.Image(res_Imgs, caption="Image")]}) ### concertar esse negocio
                    #save_path =save_dir+ config.data_name+'_level'+str(i)+'.png'
                    save_path =save_dir+ name +'_.png'
                    print("Image saved in: ",save_path)
                    cv2.imwrite(save_path, res_Imgs*255)
                
                #Metrics

                # # Wandb logs 
                # wandb.log({"Inferecia "+config.dataset:{
                #     "Test from epoch": epoch,
                #     "Image Ajuste ":wout
                #     }})

                

                



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
    Test(config,1000)
    # wandb.finish()
    #Test_for_one(modelConfig,epoch=14000)
