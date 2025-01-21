
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import os
from Loss.loss import *

# Adiciona o diretório pai ao sys.path
from Loss.loss import *
from typing import Dict
from tensorboardX import SummaryWriter
import cv2

from kornia.losses import ssim_loss
import numpy as np
import lpips

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, Pre_train=None, perceptual='DINO'):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        ### Losses
        self.L_color = None
        self.num = 0
        ### Necessita de avaliacao perceptual
        if perceptual == 'vgg':
            self.loss_perceptual = lpips.LPIPS(net='vgg')
        elif perceptual == 'squeeze':
            self.loss_perceptual = lpips.LPIPS(net='squeeze')
        elif perceptual == 'alex':
            self.loss_perceptual = lpips.LPIPS(net='alex')
        elif self.loss_perceptual == 'DINO':
            self.loss_perceptual = PerceptualLoss_dino(model="dinov2_vits14")

        self.color_loss = color_loss()
        self.L_color = L_color()
        # Oforward pass precisa de um parametro dinamico que retorne as funcoes de perda 
        # para realce e para a geracao dependendo desta flag, 
        # no segundo caso sao adicionadas as funcoes de realce relacionadas
    def forward(self, gt_images, lowlight_image, epoch):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(gt_images.shape[0],), device=gt_images.device)
        noise = torch.randn_like(gt_images)
        y_t = (
                extract(self.sqrt_alphas_bar, t, gt_images.shape) * gt_images +
                extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape) * noise)

        input = torch.cat([lowlight_image, y_t], dim=1).float()

        if torch.rand(1) < 0.02:
            noise_pred = self.model(input, t,context_zero=True)
        else:
            noise_pred = self.model(input, t)
        #########################
        ### LOSS ###Ehancement###
        #########################

        loss = 0
        mse_loss = F.mse_loss(noise_pred, noise, reduction='none')
        loss += mse_loss

        y_0_pred = 1 / extract(self.sqrt_alphas_bar, t, gt_images.shape) * (
                    y_t - extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape) * noise_pred).float()

        second_start = 20
        col_loss = 0
        col_loss_weight = 100
        if epoch < second_start:
            col_loss_weight = 0
        col_loss = self.color_loss(y_0_pred, gt_images) * col_loss_weight
        loss += col_loss

        exposure_loss = 0
        exposure_loss_weight = 20
        if epoch < second_start:
            exposure_loss_weight = 0
        exposure_loss = self.light_loss(y_0_pred, gt_images) * exposure_loss_weight
        loss += exposure_loss

        ssimLoss = 0
        ssim_weight = 2.83
        ssimLoss = ssim_loss(y_0_pred, gt_images, window_size=11) * ssim_weight
        loss += ssimLoss

        perceptual_loss = 0
        perceptual_loss_weight = 30
        perceptual_loss = self.loss_perceptual(gt_images, y_0_pred) * perceptual_loss_weight
        loss += perceptual_loss

        #Perceptual Loss


        #Color Loss


        #Generative Loss


        #Exposure Loss


        return [loss, mse_loss, col_loss, exposure_loss, ssimLoss, perceptual_loss]


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.sqrt_alphas_bar = alphas_bar
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
        self.alphas_bar = alphas_bar
        self.one_minus_alphas_bar = (1. - alphas_bar)
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, t, eps, y_t):
        assert y_t.shape == eps.shape
        return (
            extract(self.coeff1, t, y_t.shape) * y_t -
            extract(self.coeff2, t, y_t.shape) * eps
        )

    def p_mean_variance(self, input, t, y_t):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, input.shape)
        eps = self.model(input, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(t, eps, y_t)

        return xt_prev_mean, var

    def forward(self, lowlight_image, ddim=False, unconditional_guidance_scale=1, ddim_step=None):
        if ddim == False:
            device = lowlight_image.device
            noise = torch.randn_like(lowlight_image).to(device)
            y_t = noise
            for time_step in reversed(range(self.T)):
                t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step
                input = torch.cat([lowlight_image, y_t], dim=1).float()
                mean, var = self.p_mean_variance(input, t, y_t)
                if time_step > 0:
                    noise = torch.randn_like(y_t)
                else:
                    noise = 0
                y_t = mean + torch.sqrt(var) * noise

            y_0 = y_t
            return torch.clip(y_0, -1, 1)

        else:
            device = lowlight_image.device
            noise = torch.randn_like(lowlight_image).to(device)
            y_t = noise

            step = 1000 / ddim_step
            step = int(step)
            seq = range(0, 1000, step)
            seq_next = [-1] + list(seq[:-1])
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(y_t.shape[0]) * i).to(device).long()
                next_t = (torch.ones(y_t.shape[0]) * j).to(device).long()
                at = extract(self.alphas_bar.to(device), (t + 1).long(), y_t.shape)
                at_next = extract(self.alphas_bar.to(device), (next_t + 1).long(), y_t.shape)
                input = torch.cat([lowlight_image, y_t], dim=1).float()
                eps = self.model(input, t)

                # CLASSIFIER FREE GUIDANCE
                if unconditional_guidance_scale != 1:
                    eps_unconditional = self.model(input, t, context_zero=True)
                    eps = eps_unconditional + unconditional_guidance_scale * (eps - eps_unconditional)

                y0_pred = (y_t - eps * (1 - at).sqrt()) / at.sqrt()
                eta = 0
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                y_t = at_next.sqrt() * y0_pred + c1 * torch.randn_like(lowlight_image) + c2 * eps
            y_0 = y_t
            return torch.clip(y_0, -1, 1)





################# Old CODE #################
# def extract(v, t, x_shape):
#     """
#     Extract some coefficients at specified timesteps, then reshape to
#     [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
#     """
#     device = t.device
#     out = torch.gather(v, index=t, dim=0).float().to(device)
#     return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# class GaussianDiffusionTrainer(nn.Module):
#     def __init__(self, model, beta_1, beta_T, T):
#         super().__init__()

#         self.model = model
#         self.T = T

#         self.register_buffer(
#             'betas', torch.linspace(beta_1, beta_T, T).double())
#         alphas = 1. - self.betas
#         alphas_bar = torch.cumprod(alphas, dim=0)

#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer(
#             'sqrt_alphas_bar', torch.sqrt(alphas_bar))
#         self.register_buffer(
#             'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

#     def forward(self, x_0):
#         """
#         Algorithm 1.
#         """
#         t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
#         noise = torch.randn_like(x_0)
#         x_t = (
#             extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
#             extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
#         loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
#         return loss


# class GaussianDiffusionSampler(nn.Module):
#     def __init__(self, model, beta_1, beta_T, T):
#         super().__init__()

#         self.model = model
#         self.T = T

#         self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
#         alphas = 1. - self.betas
#         alphas_bar = torch.cumprod(alphas, dim=0)
#         alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

#         self.register_buffer('coeff1', torch.sqrt(1. / alphas))
#         self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

#         self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

#     def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
#         assert x_t.shape == eps.shape
#         return (
#             extract(self.coeff1, t, x_t.shape) * x_t -
#             extract(self.coeff2, t, x_t.shape) * eps
#         )

#     def p_mean_variance(self, x_t, t):
#         # below: only log_variance is used in the KL computations
#         var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
#         var = extract(var, t, x_t.shape)

#         eps = self.model(x_t, t)
#         xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

#         return xt_prev_mean, var

#     def forward(self, x_T):
#         """
#         Algorithm 2.
#         """
#         x_t = x_T
#         for time_step in reversed(range(self.T)):
#             print(time_step)
#             t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
#             mean, var= self.p_mean_variance(x_t=x_t, t=t)
#             # no noise when t == 0
#             if time_step > 0:
#                 noise = torch.randn_like(x_t)
#             else:
#                 noise = 0
#             x_t = mean + torch.sqrt(var) * noise
#             assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
#         x_0 = x_t
#         return torch.clip(x_0, -1, 1)   



# Funções de teste
class DummyModel(nn.Module):
    def forward(self, x, t, context_zero=False):
        return x[:, :3, :, :]  # Modelo fictício que retorna a entrada com o tamanho correto


def test_trainer():
    model = DummyModel()  # Modelo de teste
    trainer = GaussianDiffusionTrainer(model, 0.1, 0.2, 1000)
    gt_images = torch.randn(8, 3, 256, 256)
    epoch = 10
    [loss, mse_loss, col_loss, exp_loss, ssim_loss, perceptual_loss] = trainer(gt_images, gt_images, epoch)

    print("Trainer losses:", "Geral",loss.mean().item(), "MSE",mse_loss.mean().item(), "col",col_loss.mean().item(), "Exposure",exp_loss.mean().item(), "SSIM",ssim_loss.mean().item(), "VGG",perceptual_loss.mean().item())


def test_sampler():
    model = DummyModel()  # Modelo de teste
    sampler = GaussianDiffusionSampler(model, 0.1, 0.2, 1000)
    lowlight_image = torch.randn(8, 3, 256, 256)
    generated_image = sampler(lowlight_image)
    print("Generated image shape:", generated_image.shape)


if __name__ == "__main__":
    test_trainer()
    test_sampler()