import os
import wandb
from utils.rotinas import train, test, inference
import argparse
import pprint

def main(model_config = None):
    ## Mudar parametros para o modelo que irei usar. Utilizar parse para carregar o m etodo. Ajustar Wandb para uso e plot dos reultados 
    modelConfig = {
        "DDP": False,
        #"state": "train", # or eval
        "supervised": True,
        #"underwater_dataset_name": "UIEB",
        #"atmospheric_dataset_name": "HDR+",
        #"epoch": 2000,
        #"batch_size": 16,
        "T": 1000,
        "channel": 128,
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
        "device": "cuda:0", #MODIFIQUEI
        "device_list": [0],#[0, 1]
        #"device_list": [3,2,1,0],
        
        "ddim":True,
        "unconditional_guidance_scale":1,
        "ddim_step":100
        }
    
    ## COLOCAR OS PARAMETROS DE TREINAMENTO E TESTE AQUI
    ##Adicionar ao arg parse o transfer learning manual 
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento/Inferência do modelo Hybrid Diffusion")
    parser.add_argument('--underwater_data_name', type=str, default="HICRD")
    parser.add_argument('--atmospheric_data_name', type=str, default="LoLI")
    parser.add_argument('--model', type=str, default="standart")
    parser.add_argument('--dataset_path', type=str, default="./data/")#mudar
    parser.add_argument('--state', type=str, default="train")  #or eval
    parser.add_argument('--pretrained_path', type=str, default=None)  #or eval output/ckpt/ckpt_1000_final_UIEBTM-DIED.pt
    parser.add_argument('--inference_image', type=str, default="")  #or eval
    parser.add_argument('--output_path', type=str, default="./output/")  #or eval
    parser.add_argument('--wandb', action='store_true', help='Actavete Wandb for logging and visualization')  
    parser.add_argument('--wandb_name', type=str, default="HybridDffusion_4_ICAR")  #or eval
    parser.add_argument('--epoch', type=int, default=int(2000))
    parser.add_argument('--batch_size', type=int, default=int(16))
    parser.add_argument('--DDP', action='store_true', help="Use Distributed Data Parallel (DDP) for training")
    parser.add_argument('--stage', type=int, default=int(0))#etapa 1 e 2 paras aprendizado de caracteristicas 3 para realce de imagem
    parser.add_argument('--epochs_stage_1', type=int, default=int(1000))
    parser.add_argument('--epochs_stage_2', type=int, default=int(1000))
    parser.add_argument('--device', type=str, default=str("cuda"))
    #parser.add_argument('--DDP', type=bool, default=)

    config = parser.parse_args()
    
    # ================================
    # 4. Aplicar valores de modelConfig
    # ================================
    for key, value in modelConfig.items():
        # Só aplica se não estiver nos args (ex: não sobrescreve se foi passado via linha de comando)
        if not hasattr(config, key):
            setattr(config, key, value)

    
    
    # ================================
    # 6. Exibir os parâmetros finais
    # ================================
    print("\n🔧 Configurações Finais:")
    pprint.pprint(vars(config))

    if config.wandb:
        # Verifica se o arquivo de token do wandb existe
        with open('wandb_token.txt', 'r') as f:
            token = f.read().strip()

        # Define o token como variável de ambiente
        os.environ['WANDB_API_KEY'] = token
        wandb.init(
                project=config.wandb_name,
                config=vars(config),
                name= config.state +"_"+ config.wandb_name +"_"+ config.underwater_data_name+"_"+config.atmospheric_data_name,
                tags=[config.state, config.underwater_data_name, config.atmospheric_data_name],
                group="HybridDiffusion",
                job_type="train"

            ) 


    ##################################################
    ### Treinamento, Teste, Avaliação e Inferência ###
    ##################################################
    if config.state == 'eval':
        print("Avaliando modelo")
        inference(config, config.epoch)
    elif config.state == 'train':
        print("Treinando modelo")
        train(config)
    elif config.state == 'inference':
        print("Inferindo modelo")
        test(config,config.epoch)
    else:
        print("Invalid state/nPlease use 'train', 'eval' or 'inference'.")
    #train(config)#importar a funcao ou classe de papeline de treinamento== treino/teste e carregar as configs e rodar
    #Testi(config, 1000)
    #python main.py --dataset "RUIE" --state "test" --epoch 500
    #python main.py --dataset "UIEB" --state "train" --epoch 1000

    if config.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
