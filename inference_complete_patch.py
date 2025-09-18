"""
inference_complete_patch.py

This file provides a complete `Inference` function to patch:
https://github.com/gusanagy/Diffusion-Underwater-Atmopheric-Image-Enhancement

Purpose:
The repository currently lacks a working inference function. This patch completes it,
mainly based on the original author's draft, with minor additions.

Usage:
Copy the `Inference` function from this file into `utils/rotinas.py` of the original repo.
It is not a standalone script — it is meant to be inserted where inference logic is missing.

Note:
- The logic stays close to the original repo.
- Output quality may vary depending on model weights.
- Some evaluation code may still need fixes — working on that next.

Author: A beginner who loves coding
Date: 2025-04-13
Contact :Actually, I sent you a private message on ins, but you ignored me and was so sad/(ㄒoㄒ)/~~
---

Olá! Desculpe incomodar. Eu não falo português, esta mensagem foi traduzida automaticamente.
Este patch é baseado no seu excelente projeto. Só fiz pequenas adições onde原始项目缺少推理函数。
Obrigado pelo trabalho incrível! Espero :)
"""

def Inference(config: Dict, epoch):
    """
    Core inference process for diffusion model

    Parameters:
    config: Configuration dictionary
    epoch: Current epoch number

    Steps:
    1. Data loading and preprocessing
    2. Model loading and configuration
    3. Diffusion sampling process
    4. Result post-processing and saving
    """
    # Load inference data
    datapath_test = [config.inference_image]
    device = config.device_list[0]

    # Data augmentation configuration
    transform = A.Compose([A.Resize(256, 256),
                           ToTensorV2()])

    # Initialize underwater dataset
    underwater_data = Underwater_Dataset(
        underwater_dataset_name="LSUI",
        transforms=transform,
        task="inference",
        supervised=config.supervised,
    )

    # Set path parameters
    underwater_data.train_img_u = datapath_test  # Set image paths
    underwater_data.test_img_u = datapath_test
    underwater_data.val_img_u = datapath_test
    dataload_test = underwater_data
    print("🔍 Inference image path:", underwater_data.test_img_u)

    # Create dataloader
    dataloader = DataLoader(dataload_test, batch_size=1, num_workers=0, drop_last=True, pin_memory=True)
    print("✅ Starting inference, total images:", len(dataloader))

    # Initialize model
    model = DynamicUNet(
        T=config.T,
        ch=config.channel,
        ch_mult=config.channel_mult,
        num_res_blocks=config.num_res_blocks,
        dropout=config.dropout
    ).to(device)

    # Load model weights
    ckpt = torch.load(config.pretrained_path, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    print("Model weights loaded successfully.")

    # Create save directory
    save_dir = os.path.join(config.output_path, "result", os.path.basename(config.pretrained_path),
                            config.underwater_dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize sampler
    model.eval()
    sampler = GaussianDiffusionSampler(model, config.beta_1, config.beta_T, config.T).to(device)

    # Inference loop
    with torch.no_grad():
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for idx, batch in enumerate(tqdmDataLoader):
                # Parse batch data
                if config.supervised:
                    input_image, gt_image, filename = batch
                    name = filename[0].split("/")[-1]
                else:
                    input_image, gt_image, filename = batch
                    name = f"infer_{idx:04d}.png"

                input_image = input_image.to(device)

                # Diffusion sampling process
                sampledImgs = sampler(input_image, ddim=True,
                                      unconditional_guidance_scale=1,
                                      ddim_step=config.ddim_step)

                # Post-processing
                sampledImgs = (sampledImgs + 1) / 2.0
                res_Imgs = sampledImgs.detach().cpu().numpy()[0]  # Shape: (3, H, W)

                print("Sampled images shape:", sampledImgs.shape)

                # Print channel statistics
                for i, color in enumerate(['R', 'G', 'B']):
                    channel = res_Imgs[i]
                    print(f"{color} mean: {channel.mean():.4f}, max: {channel.max():.4f}, min: {channel.min():.4f}")

                # Convert and normalize
                res_Imgs = np.clip(sampledImgs.detach().cpu().numpy()[0].transpose(1, 2, 0), 0, 1)
                print("Sampled images min/max:", sampledImgs.min().item(), sampledImgs.max().item())
                print("Raw statistics:", sampledImgs.min(), sampledImgs.max(), sampledImgs.mean())

                # Save results
                save_path = os.path.join(save_dir, name)
                print("Saving image to:", save_path)
                # Save directly in RGB format without converting to BGR
                cv2.imwrite(save_path, (res_Imgs * 255).astype(np.uint8))
                np.save('debug_output.npy', sampledImgs.detach().cpu().numpy())
