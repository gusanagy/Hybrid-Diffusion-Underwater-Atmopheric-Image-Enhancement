from Diffusion.Train import train, eval
import wandb
from utils.rotinas import *
import argparse


def main(model_config = None):
    ## Mudar parametros para o modelo que irei usar. Utilizar parse para carregar o m etodo. Ajustar Wandb para uso e plot dos reultados 
    modelConfig = {
        "state": "train", # or eval
        "supervised": False,
        "underwater_dataset_name": "UIEB",
        "atmospheric_dataset_name": "HDR+",
        "epoch": 200,
        "batch_size": 80,
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
        "device_list": [0, 1],#[0, 1]
        #"device_list": [3,2,1,0],
        
        "ddim":True,
        "unconditional_guidance_scale":1,
        "ddim_step":100
        }
    
    ## COLOCAR OS PARAMETROS DE TREINAMENTO E TESTE AQUI
    ##Adicionar ao arg parse o transfer learning manual para o mask diffusions
    parser.add_argument('--dataset', type=str, default="all")
    parser.add_argument('--model', type=str, default="standart")#mask is the second option
    parser.add_argument('--dataset_path', type=str, default="./data/UDWdata/")
    parser.add_argument('--state', type=str, default="train")  #or eval
    parser.add_argument('--pretrained_path', type=str, default=None)  #or eval
    parser.add_argument('--inference_image', type=str, default="data/UDWdata/UIEB/val/206_img_.png")  #or eval
    parser.add_argument('--output_path', type=str, default="./output/")  #or eval
    parser.add_argument('--wandb', type=bool, default=False)  #or False
    parser.add_argument('--wandb_name', type=str, default="HybridDffusion")
    parser.add_argument('--epoch', type=int, default=int(1000))
    #adicionar mais argumentos para o wandb

    config = parser.parse_args()
    
    # if config.wandb:
    #     wandb.init(
    #             project=config.wandb_name,
    #             config=vars(config),
    #             name= config.state +"_"+ config.wandb_name +"_"+ config.dataset,
    #             tags=[config.state, config.dataset],
    #             group="Branch glown_diffusion_test",
    #             job_type="test"

    #         ) 
    
    for key, value in modelConfig.items():
        setattr(config, key, value)
    
    print(config)

    print(config.epoch)

    if config.state == 'eval':
        Test(config, config.epoch)
    elif config.state == 'train':
        train(config)
    elif config.state == 'inference':
        Inference(config,config.epoch)
    else:
        print("Invalid state")
    #train(config)#importar a funcao ou classe de papeline de treinamento== treino/teste e carregar as configs e rodar
    #Testi(config, 1000)
    #python main.py --dataset "RUIE" --state "test" --epoch 500
    #python main.py --dataset "UIEB" --state "train" --epoch 1000

    # if config.wandb:
    #     wandb.finish()

if __name__ == '__main__':
    main()
