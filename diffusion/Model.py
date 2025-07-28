
   
import math
from telnetlib import PRAGMA_HEARTBEAT
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class ConditionalEmbedding_old(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        num_labels = 2 if num_labels is None else num_labels
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.condEmbedding(t)
        return emb

# class ConditionalEmbedding(nn.Module):
#     def __init__(self, d_model, dim):
#         super().__init__()

#         # Camadas convolucionais para reduzir a resolução da imagem
#         # self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)      # Reduz a resolução
#         # self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)    # Mais redução
#         # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)   # Mais redução

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)      # Reduzido
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)     # Reduzido
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)    # Reduzido
#         # Pooling global para reduzir para 1x1 em cada canal
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling global

#         # Camada linear
#         self.linear1 = nn.Linear(256, dim)
#         self.activation = Swish()
#         self.linear2 = nn.Linear(dim, dim)

#     def forward(self, label_tensor):
#         """
#         Processa a imagem de rótulo para gerar um embedding condicional.

#         Args:
#         - label_tensor: Tensor da imagem de rótulo [B, C, H, W].

#         Retorno:
#         - emb: Embedding condicional baseado na imagem de rótulo.
#         """
#         # Passa pelas camadas convolucionais
#         x = self.conv1(label_tensor)  # [B, 64, H/2, W/2]
#         x = self.conv2(x)  # [B, 128, H/4, W/4]
#         x = self.conv3(x)  # [B, 256, H/8, W/8]

#         # Aplica pooling para reduzir para 1x1 em cada canal
#         x = self.pool(x)  # [B, 256, 1, 1]

#         # Flatten para [B, 256]
#         x = x.view(x.size(0), -1)  # [B, 256]

#         # Passa pelas camadas lineares
#         emb = self.linear1(x)
#         emb = self.activation(emb)
#         emb = self.linear2(emb)

#         return emb
    
class ConditionalEmbedding(nn.Module):
    def __init__(self, d_model, dim):
        super().__init__()

        # Calculando os canais intermediários com base no d_model
        channels = d_model // 16  # Ajustando os canais de acordo com d_model

        # Camadas convolucionais para reduzir a resolução da imagem
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, stride=2, padding=1)  # Reduz a resolução
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=3, stride=2, padding=1)  # Mais redução
        self.conv3 = nn.Conv2d(in_channels=channels*2, out_channels=channels*4, kernel_size=3, stride=2, padding=1)  # Mais redução

        # # Camadas de deconvolução (ConvTranspose2d) para reconstruir a imagem
        # self.deconv1 = nn.ConvTranspose2d(in_channels=channels*4, out_channels=channels*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv2 = nn.ConvTranspose2d(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv3 = nn.ConvTranspose2d(in_channels=channels, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Pooling global para reduzir para 1x1 em cada canal
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling global

        # Camada linear
        self.linear1 = nn.Linear(channels*4, dim)  # Alterando para usar o valor de canais após as convoluções
        self.activation = Swish()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, label_tensor):
        """
        Processa a imagem de rótulo para gerar um embedding condicional.

        Args:
        - label_tensor: Tensor da imagem de rótulo [B, C, H, W].

        Retorno:
        - emb: Embedding condicional baseado na imagem de rótulo.
        """
        # Passa pelas camadas convolucionais
        x = self.conv1(label_tensor)  # [B, channels, H/2, W/2]
        x = self.conv2(x)  # [B, channels*2, H/4, W/4]
        x = self.conv3(x)  # [B, channels*4, H/8, W/8]

        # Aplica pooling para reduzir para 1x1 em cada canal
        x = self.pool(x)  # [B, channels*4, 1, 1]

        # Flatten para [B, channels*4]
        x = x.view(x.size(0), -1)  # [B, channels*4]

        # Passa pelas camadas lineares
        emb = self.linear1(x)
        emb = self.activation(emb)
        emb = self.linear2(emb)

        # Passa pelas camadas de deconvolução (opcional dependendo da aplicação)
        x = emb.view(emb.size(0), -1, 1, 1)  # Reshape para imagem com 1x1
        # x = self.deconv1(x)  # [B, channels*2, H/4, W/4]
        # x = self.deconv2(x)  # [B, channels, H/2, W/2]
        # x = self.deconv3(x)  # [B, 3, H, W] (imagem reconstruída)

        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb, cemb):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x


class AttnBlock_old(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock_old(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()


    def forward(self, x, temb, labels):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h += self.cond_proj(labels)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super(ResBlock,self).__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        self.activate_attn = attn
        self.attn = nn.MultiheadAttention(out_ch, num_heads=8) if attn else nn.Identity()

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb, cemb=None):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        if cemb is not None:
            h += self.cond_proj(cemb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        
        if self.activate_attn:#Posso ter entendido mal a aplicacao do attention
            batch, channels, height, width = h.shape
            h_reshaped = h.view(batch, channels, -1).permute(2, 0, 1)  # (seq_len, batch, channels)
            h_attn, _ = self.attn(h_reshaped, h_reshaped, h_reshaped)
            h = h_attn.permute(1, 2, 0).view(batch, channels, height, width)

        return h

#class UNet(nn.Module):
    # def __init__(self, T, num_labels, ch, ch_mult, num_res_blocks, dropout):
    #     super().__init__()
    #     tdim = ch * 4
    #     num_labels = 2
    #     self.time_embedding = TimeEmbedding(T, ch, tdim)
    #     self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)
    #     self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
    #     self.downblocks = nn.ModuleList()
    #     chs = [ch]  # record output channel when dowmsample for upsample
    #     now_ch = ch
    #     for i, mult in enumerate(ch_mult):
    #         out_ch = ch * mult
    #         for _ in range(num_res_blocks):
    #             self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
    #             now_ch = out_ch
    #             chs.append(now_ch)
    #         if i != len(ch_mult) - 1:
    #             self.downblocks.append(DownSample(now_ch))
    #             chs.append(now_ch)

    #     self.middleblocks = nn.ModuleList([
    #         ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
    #         ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
    #     ])

    #     self.upblocks = nn.ModuleList()
    #     for i, mult in reversed(list(enumerate(ch_mult))):
    #         out_ch = ch * mult
    #         for _ in range(num_res_blocks + 1):
    #             self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
    #             now_ch = out_ch
    #         if i != 0:
    #             self.upblocks.append(UpSample(now_ch))
    #     assert len(chs) == 0

    #     self.tail = nn.Sequential(
    #         nn.GroupNorm(32, now_ch),
    #         Swish(),
    #         nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
    #     )
 

    # def forward(self, x, t, labels):
    #     # Timestep embedding
    #     temb = self.time_embedding(t)
    #     cemb = self.cond_embedding(labels)
    #     # Downsampling
    #     h = self.head(x)
    #     hs = [h]
    #     for layer in self.downblocks:
    #         h = layer(h, temb, cemb)
    #         hs.append(h)
    #     # Middle
    #     for layer in self.middleblocks:
    #         h = layer(h, temb, cemb)
    #     # Upsampling
    #     for layer in self.upblocks:
    #         if isinstance(layer, ResBlock):
    #             h = torch.cat([h, hs.pop()], dim=1)
    #         h = layer(h, temb, cemb)
    #     h = self.tail(h)

    #     assert len(hs) == 0
    #     return h
    


class DynamicUNet(nn.Module):
    def __init__(self, T, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        #self.attn = attn
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = ConditionalEmbedding(ch, tdim)

        ## Layers
        self.head = nn.Conv2d(6, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks, self.chs, self.now_ch = self.create_downblocks(ch, ch_mult, num_res_blocks, tdim, dropout)
        self.middleblocks = self.create_middleblocks(self.now_ch, tdim, dropout)
        self.upblocks = self.create_upblocks(ch, ch_mult, num_res_blocks, tdim, dropout)

        self.tail = nn.Sequential(
            nn.GroupNorm(32, ch),
            Swish(),
            nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def create_downblocks(self, ch, ch_mult, num_res_blocks, tdim, dropout):
        downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
        return downblocks, chs, now_ch

    def create_middleblocks(self, now_ch, tdim, dropout):
        return nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True)
        ])

    def create_upblocks(self, ch, ch_mult, num_res_blocks, tdim, dropout):
        upblocks = nn.ModuleList()
        chs = self.chs.copy()
        now_ch = self.now_ch

        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                upblocks.append(UpSample(now_ch))
        return upblocks
    def dynamic_forward(self, x):
        """
        Dynamic forward pass based on input image type.
        x: input image tensor
        t: timestep embedding tensor
        labels: conditional labels tensor
        """
        # Determine image type using statistical thresholding on color channels
        red_channel_mean = x[:, 0, :, :].mean()  # Mean of red channel
        blue_channel_mean = x[:, 2, :, :].mean()  # Mean of blue channel

        # Activate subaquatic layers if red < blue; otherwise, activate atmospheric layers
        is_subaquatic = blue_channel_mean > red_channel_mean

        for i, layer in enumerate(self.middleblocks):
            if is_subaquatic:
                if i % 2 == 0:  # Enable subaquatic layers (even-indexed)
                    for param in layer.parameters():
                        param.requires_grad = True
                else:  # Freeze atmospheric layers (odd-indexed)
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                if i % 2 != 0:  # Enable atmospheric layers (odd-indexed)
                    for param in layer.parameters():
                        param.requires_grad = True
                else:  # Freeze subaquatic layers (even-indexed)
                    for param in layer.parameters():
                        param.requires_grad = False
    def forward(self, x, t, labels=None, context_zero=True):
        # Ajuste dinâmico dos middleblocks com base no tipo de entrada
        self.dynamic_forward(x)
        #Time embedding from Diffusion
        temb = self.time_embedding(t)
        #print(f"Labels shape{labels.shape, labels.dtype}")

        #No classifier Guidance Part
        if context_zero:
            cemb = torch.zeros_like(temb)
        else:
            cemb = self.cond_embedding(labels)
        #print(f"Labels shape{x.shape, x.dtype}")
        # Downsampling
        h = self.head(x)
        hs = [h]  # Armazena saída inicial
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)  # Salva a saída de cada camada

        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, cemb)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                # Garante que sempre existe um tensor para concatenar
                assert len(hs) > 0, "A lista `hs` está vazia antes do esperado."

                # Remove o último tensor de `hs` e ajusta o tamanho, se necessário
                skip_h = hs.pop()
                if h.shape[2:] != skip_h.shape[2:]:
                    skip_h = F.interpolate(skip_h, size=h.shape[2:], mode="nearest")

                h = torch.cat([h, skip_h], dim=1)
            h = layer(h, temb, cemb)

        # # Garantia final
        # print(f"Elementos restantes em hs: {len(hs)}")
        # assert len(hs) == 0, f"A lista `hs` contém {len(hs)} elementos restantes no final do Upsampling."

        return self.tail(h)

#class DynamicUNet_0(nn.Module):
    # def __init__(self, T, ch, ch_mult, num_res_blocks, dropout, attn):
    #     super().__init__()
    #     self.attn = attn
    #     tdim = ch * 4
    #     num_labels = 2
    #     self.time_embedding = TimeEmbedding(T, ch, tdim)
    #     self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)

    #     ## Layers 
    #     self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
    #     self.downblocks, self.chs, self.now_ch = self.create_downblocks(ch, ch_mult, num_res_blocks, tdim, dropout)
    #     self.middleblocks = self.create_middleblocks(self.now_ch, tdim, dropout)
    #     self.upblocks = self.create_upblocks(ch, ch_mult, num_res_blocks, tdim, dropout)
        
    #     self.tail = nn.Sequential(
    #         nn.GroupNorm(32, 128),
    #         Swish(),
    #         nn.Conv2d(self.now_ch, 3, 3, stride=1, padding=1)
    #     )
    #     self.initialize()

    # def initialize(self):
    #     init.xavier_uniform_(self.head.weight)
    #     init.zeros_(self.head.bias)
    #     init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
    #     init.zeros_(self.tail[-1].bias)


    # def create_downblocks(self, ch, ch_mult, num_res_blocks, tdim, dropout):
    #     downblocks = nn.ModuleList()
    #     chs = [ch]
    #     now_ch = ch

    #     for i, mult in enumerate(ch_mult):
    #         out_ch = ch * mult
    #         for _ in range(num_res_blocks):
    #             downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout,attn=False))
    #             now_ch = out_ch
    #             chs.append(now_ch)
    #         if i != len(ch_mult) - 1:
    #             downblocks.append(DownSample(now_ch))
    #             chs.append(now_ch)
    #     print("now channel", chs)
    #     return downblocks, chs, now_ch

    # def create_middleblocks(self, now_ch, tdim, dropout):
    #     return nn.ModuleList([
    #         ResBlock(now_ch, now_ch, tdim, dropout, attn=True),# Subaquatic
    #         ResBlock(now_ch, now_ch, tdim, dropout, attn=False), # Atmospheric
    #         ResBlock(now_ch, now_ch, tdim, dropout, attn=True),# Subaquatic
    #         ResBlock(now_ch, now_ch, tdim, dropout, attn=False),# Atmospheric
    #     ])

    # def create_upblocks(self, ch, ch_mult, num_res_blocks, tdim, dropout):
    #     upblocks = nn.ModuleList()
    #     chs = self.chs.copy()
    #     now_ch = self.now_ch

    #     for i, mult in reversed(list(enumerate(ch_mult))):
    #         out_ch = ch * mult
    #         for _ in range(num_res_blocks + 1):
    #             upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
    #             now_ch = out_ch
    #         if i != 0:
    #             upblocks.append(UpSample(now_ch))
    #     print('Now_ch', chs)
    #     assert len(chs) == 0
    #     return upblocks

    # def dynamic_forward(self, x, t, labels):
    #     """
    #     Dynamic forward pass based on input image type.
    #     x: input image tensor
    #     t: timestep embedding tensor
    #     labels: conditional labels tensor
    #     """
    #     # Determine image type using statistical thresholding on color channels
    #     red_channel_mean = x[:, 0, :, :].mean()  # Mean of red channel
    #     blue_channel_mean = x[:, 2, :, :].mean()  # Mean of blue channel

    #     # Activate subaquatic layers if red < blue; otherwise, activate atmospheric layers
    #     is_subaquatic = blue_channel_mean > red_channel_mean

    #     for i, layer in enumerate(self.middleblocks):
    #         if is_subaquatic:
    #             if i % 2 == 0:  # Enable subaquatic layers (even-indexed)
    #                 for param in layer.parameters():
    #                     param.requires_grad = True
    #             else:  # Freeze atmospheric layers (odd-indexed)
    #                 for param in layer.parameters():
    #                     param.requires_grad = False
    #         else:
    #             if i % 2 != 0:  # Enable atmospheric layers (odd-indexed)
    #                 for param in layer.parameters():
    #                     param.requires_grad = True
    #             else:  # Freeze subaquatic layers (even-indexed)
    #                 for param in layer.parameters():
    #                     param.requires_grad = False
    # def forward(self, x, t, labels):
    #     # Ajuste dinâmico dos middleblocks com base no tipo de entrada
    #     self.dynamic_forward(x, t, labels)

    #     # Timestep embedding
    #     temb = self.time_embedding(t)
    #     cemb = self.cond_embedding(labels)
        
    #     # Downsampling
    #     h = self.head(x)
    #     hs = [h]
    #     for layer in self.downblocks:
    #         h = layer(h, temb, cemb)  # Passando os argumentos corretos
    #         hs.append(h)

    #     # Middle
    #     for layer in self.middleblocks:
    #         h = layer(h, temb, cemb)  # Passando os argumentos corretos

    #     # Upsampling
    #     for layer in self.upblocks:
    #         if isinstance(layer, ResBlock):
    #             h = torch.cat([h, hs.pop()], dim=1)
    #         h = layer(h, temb, cemb)  # Passando os argumentos corretos
        
    #     print(f"Input tail Block{h.shape}")

    #     h = self.tail(h)
    #     #assert len(hs) == 0
    #     return h
    

if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randint(10, size=[batch_size])
    # resB = ResBlock(128, 256, 64, 0.1)
    # x = torch.randn(batch_size, 128, 32, 32)
    # t = torch.randn(batch_size, 64)
    # labels = torch.randn(batch_size, 64)
    # y = resB(x, t, labels)
    y = model(x, t, labels)
    print(y.shape)

