import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.models as models
from kornia.losses import ssim_loss as ssim, psnr_loss as psnr, MS_SSIMLoss as ms_ssim, charbonnier_loss as charbonnier


#############################################################
###Funcoes para aprenzado de caracteristicas dos datasets ###
#############################################################


## DINO LOSS FUNCTIONS
class PerceptualLoss_dino(nn.Module):
    def __init__(self, model:str="dinov2_vits14",version:str ="dinov2", layers=None, normalize_inputs=False):
        """
        Perceptual loss using DINO or DINOv2 models.

        Args:
            model (torch.nn.Module): The DINO model to extract features.
            layers (list of str): Names of layers to use for loss computation.
                                  Default is None, which uses all layers.
            normalize_inputs (bool): Whether to normalize inputs to [0, 1].
        """
        super(PerceptualLoss_dino, self).__init__()

        # Load DINO model
        if version == "dino":
            self.model = torch.hub.load('facebookresearch/dino:main', model)
        elif version == "dinov2":
            self.model = torch.hub.load('facebookresearch/dinov2', model)
        else:
            raise ValueError(f"""Version {version} not found. Choose between 'dino' or 'dinov2'\nDino versions avaiable:
    #Dino V1
        vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8') # funcionando
        vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    # DINOv2
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

    # DINOv2 with registers
        dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')""")
        

        self.model = self.model.eval()  # Set model to evaluation mode
        self.layers = layers
        self.normalize_inputs = normalize_inputs

        # Disable gradient computation for the model
        for param in self.model.parameters():
            param.requires_grad = False
    def _crop_tensor(self, tensor):
        """
        Realiza um crop de um tensor de tamanho (1, 3, 256, 256) para (1, 3, 252, 252).
        
        Parâmetros:
            tensor (torch.Tensor): Tensor no formato (1, 3, 256, 256).
        
        Retorna:
            torch.Tensor: Tensor cropped no formato (1, 3, 252, 252).
        """
        # Calcula os índices para o corte
        crop_size = 252
        start = (256 - crop_size) // 2  # Começo do corte
        end = start + crop_size         # Fim do corte

        #tensor  = tensor.to("cuda")
        
        # Aplica o corte diretamente
        cropped_tensor = F.pad(tensor, pad=(-start, -start, -start, -start))
        #print(f"Input shape: {tensor.shape}, Output shape: {cropped_tensor.shape}")
        
        return cropped_tensor
    def _ensure_tensor(self, feat):
            if isinstance(feat, tuple):
                feat = feat[0]
            return feat
    
    def extract_features(self, x):
        """
        Extract features from the model.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).

        Returns:
            list of torch.Tensor: Extracted features.
        """
        #x = x.float() / 255.0
        features = []
        hooks = []

        # Hook to extract features from specified layers
        def hook_fn(module, input, output):
            features.append(output)

        # Register hooks on specified layers or use all layers
        if self.layers is None:
            for name, module in self.model.named_modules():
                hooks.append(module.register_forward_hook(hook_fn))
        else:
            for name, module in self.model.named_modules():
                if name in self.layers:
                    hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass to get features
        self.model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return features


    def forward(self, input, target):
        
        """
        Compute perceptual loss between input and target images.

        Args:
            input (torch.Tensor): Input image tensor (B, C, H, W).
            target (torch.Tensor): Target image tensor (B, C, H, W).

        Returns:
            torch.Tensor: Perceptual loss value.
        # """

        if self.normalize_inputs:
            input = (input - input.min()) / (input.max() - input.min())
            target = (target - target.min()) / (target.max() - target.min())

        # Extract features
        input_features = self.extract_features(self._crop_tensor(input))
        target_features = self.extract_features(self._crop_tensor(target))

        # Compute loss
        loss = 0
        for inp_feat, tgt_feat in zip(input_features, target_features):
            
            inp_feat = self._ensure_tensor(inp_feat)
            tgt_feat = self._ensure_tensor(tgt_feat)

            loss += nn.functional.smooth_l1_loss(inp_feat, tgt_feat, reduction='mean')
        return loss
    


# Definindo a PerceptualLoss
class PerceptualLoss_vgg(nn.Module):
    def __init__(self, id: int = None, model='vgg16', layer_indices=None):
        super(PerceptualLoss_vgg, self).__init__()
        # Load the model
        self._id = id
        if model == 'vgg11':
            self.perceptual = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
        elif model == 'vgg11_bn':
            self.perceptual = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1).features
        elif model == 'vgg13':
            self.perceptual = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1).features
        elif model == 'vgg13_bn':
            self.perceptual = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1).features
        elif model == 'vgg16':
            self.perceptual = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        elif model == 'vgg16_bn':
            self.perceptual = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1).features
        elif model == 'vgg19':
            self.perceptual = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        elif model == 'vgg19_bn':
            self.perceptual = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).features
        elif model == 'squeeze':
            self.perceptual = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1).features
        elif model == 'alex':
            self.perceptual = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features
        else:
            raise ValueError("Unsupported perceptual model type. Choose from ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'squeeze', 'alex']")
        
        self._model = model
        
        self.perceptual.eval()  # Set to evaluation mode
        for param in self.perceptual.parameters():
            param.requires_grad = False  # Freeze the parameters
        
        self.layer_indices = {
            'squeeze':  [3, 7, 12],
            'vgg11':    [3, 8, 15, 22],
            'vgg11_bn': [3, 8, 15, 22],
            'vgg13':    [3, 8, 15, 22],
            'vgg13_bn': [3, 8, 15, 22],
            'vgg16':    [3, 8, 15, 22],
            'vgg16_bn': [3, 8, 15, 22],
            'vgg19':    [3, 8, 17, 26, 35],
            'vgg19_bn': [3, 8, 17, 26, 35],
            'alex':     [3, 6, 8, 10, 12],
        }

        if layer_indices is not None:
            self.layer_indices[model] = layer_indices
    
    @property
    def name(self):
        return self.__class__.__name__ + '_' + self._model

    @property
    def id(self):
        return self._id
    
    def forward(self, x, y):
        # Normalize the inputs
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)#.to(x.device)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)#.to(x.device)
        # x = (x - mean) / std
        # y = (y - mean) / std

        # Extract features
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)

        # Calculate perceptual loss
        loss = 0.0
        for xf, yf in zip(x_features, y_features):
            loss += nn.functional.l1_loss(xf, yf, reduction='mean',)

        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.perceptual):
            x = layer(x)
            if i in self.layer_indices[self._model]:
                features.append(x)
        return features

###################################################
###Funcoes para o aprendizdo de realce dos dados###
###################################################

"""Angular Color Loss function"""##mudar nome %
class angular_color_loss(nn.Module):
    def __init__(self,id:int = None):
        super(angular_color_loss, self).__init__()
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id
    
    def forward(self, output, gt,mask=None):
        img_ref = F.normalize(output, p = 2, dim = 1)
        ref_p = F.normalize(gt, p = 2, dim = 1)
        loss_cos = 1 - torch.mean(F.cosine_similarity(img_ref, ref_p, dim=1))
        # loss_cos = self.mse(img_ref, ref_p)
        return loss_cos

##############################################3
 
"""MS-SSIM Loss function"""
class MSSSIMLoss(nn.Module):
    def __init__(self, id: int = None):
        super(MSSSIMLoss, self).__init__()
        self._id = id
        self.loss = ms_ssim()
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def id(self):
        return self._id

    def forward(self, input, target):
        return  self.loss(input, target)
    
"""Charbonnier Loss function"""
class CharbonnierLoss(nn.Module):
    def __init__(self, id: int = None):
        super(CharbonnierLoss, self).__init__()
        self._id = id

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def id(self):
        return self._id

    def forward(self, input, target):
        return charbonnier(input, target)