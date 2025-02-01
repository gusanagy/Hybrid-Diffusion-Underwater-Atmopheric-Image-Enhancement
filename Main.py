import wandb
from utils.rotinas import *
import argparse
import pprint


def main(model_config = None):
    ## Mudar parametros para o modelo que irei usar. Utilizar parse para carregar o m etodo. Ajustar Wandb para uso e plot dos reultados 
    modelConfig = {
        "DDP": False,
        #"state": "train", # or eval
        "supervised": True,
        "underwater_dataset_name": "UIEB",
        "atmospheric_dataset_name": "HDR+",
        "epoch": 1000,
        #"batch_size": 16,
        "T": 1000,
        "channel": 64,
        "channel_mult": [1, 2, 2, 2],
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
        "device_list": [0, 1],#[0, 1]
        #"device_list": [3,2,1,0],
        
        "ddim":True,
        "unconditional_guidance_scale":1,
        "ddim_step":100
        }
    
    ## COLOCAR OS PARAMETROS DE TREINAMENTO E TESTE AQUI
    ##Adicionar ao arg parse o transfer learning manual 
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento/Inferência do modelo Hybrid Diffusion")
    parser.add_argument('--underwater_data_name', type=str, default="HICRD")
    parser.add_argument('--atmospheric_data_name', type=str, default="TM-DIED")
    parser.add_argument('--model', type=str, default="standart")
    parser.add_argument('--dataset_path', type=str, default="./data/")
    parser.add_argument('--state', type=str, default="train")  #or eval
    parser.add_argument('--pretrained_path', type=str, default=None)  #or eval output/ckpt/ckpt_1000_final_UIEBTM-DIED.pt
    parser.add_argument('--inference_image', type=str, default="")  #or eval
    parser.add_argument('--output_path', type=str, default="./output/")  #or eval
    parser.add_argument('--wandb', type=bool, default=False)  #or False
    parser.add_argument('--wandb_name', type=str, default="HybridDffusion_2")
    parser.add_argument('--epoch', type=int, default=int(1000))
    parser.add_argument('--batch_size', type=int, default=int(16))
    parser.add_argument('--DDP', type=bool, default=False)
    parser.add_argument('--stage', type=int, default=int(0))#etapa 1 e 2 paras aprendizado de caracteristicas 3 para realce de imagem
    parser.add_argument('--epochs_stage_3', type=int, default=int(200))
    parser.add_argument('--epochs_stage_1', type=int, default=int(400))
    parser.add_argument('--epochs_stage_2', type=int, default=int(400))
    #parser.add_argument('--DDP', type=bool, default=)


    #adicionar mais argumentos para o wandb

    config = parser.parse_args()
    
    if config.wandb:
        wandb.init(
                project=config.wandb_name,
                config=vars(config),
                name= config.state +"_"+ config.wandb_name +"_"+ config.underwater_data_name+"_"+config.atmospheric_data_name,
                tags=[config.state, config.underwater_data_name, config.atmospheric_data_name],
                group="HybridDiffusion",
                job_type="train"

            ) 
    
    for key, value in modelConfig.items():
        setattr(config, key, value)
    
    pprint.pprint(config)

    print(config.epoch)

    if config.state == 'eval':
        print("Avaliando modelo")
        val(config, config.epoch)
    elif config.state == 'train':
        print("Treinando modelo")
        train(config)
    elif config.state == 'inference':
        print("Inferindo modelo")
        Inference(config,config.epoch)
    else:
        print("Invalid state")
    #train(config)#importar a funcao ou classe de papeline de treinamento== treino/teste e carregar as configs e rodar
    #Testi(config, 1000)
    #python main.py --dataset "RUIE" --state "test" --epoch 500
    #python main.py --dataset "UIEB" --state "train" --epoch 1000

    if config.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
