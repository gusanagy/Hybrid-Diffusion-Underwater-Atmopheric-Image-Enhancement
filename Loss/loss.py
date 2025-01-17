import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F



## DINO LOSS FUNCTIONS
"""
Dino versions avaiable:
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
        dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
"""

class PerceptualLoss(nn.Module):
    def __init__(self, model:str="dinov2_vits14",version:str ="dinov2", layers=None, normalize_inputs=False):
        """
        Perceptual loss using DINO or DINOv2 models.

        Args:
            model (torch.nn.Module): The DINO model to extract features.
            layers (list of str): Names of layers to use for loss computation.
                                  Default is None, which uses all layers.
            normalize_inputs (bool): Whether to normalize inputs to [0, 1].
        """
        super(PerceptualLoss, self).__init__()

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
        

        self.model = model.eval()  # Set model to evaluation mode
        self.layers = layers
        self.normalize_inputs = normalize_inputs

        # Disable gradient computation for the model
        for param in self.model.parameters():
            param.requires_grad = False

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
        input_features = self.extract_features(input)
        target_features = self.extract_features(target)

        # Compute loss
        loss = 0
        for inp_feat, tgt_feat in zip(input_features, target_features):
            
            inp_feat = self._ensure_tensor(inp_feat)
            tgt_feat = self._ensure_tensor(tgt_feat)

            loss += nn.functional.smooth_l1_loss(inp_feat, tgt_feat).to('cpu')
        return loss
