import torchvision.models as models
from torchvision import transforms

def get_model(model_name, freeze=True, verbose=True):
    """
    Dispatcher function to load models by name.
    """
    model_name = model_name.lower().strip()
    
    if model_name == "resnet50":
        model, preprocess = load_resnet50(verbose=verbose)
    elif model_name == "convnext_base":
        model, preprocess = load_convnext_base(verbose=verbose)
    elif model_name == "resnet50_v1":
        model, preprocess = load_resnet50_v1(verbose=verbose)
    elif model_name == "vit_b_16_augreg":
        model, preprocess = load_vit_b_16_augreg(verbose=verbose)
    elif model_name == "vit_b_16_augreg2":
        model, preprocess = load_vit_b_16_augreg2(verbose=verbose)
    elif model_name == "resnet18":
        model, preprocess = load_resnet18(verbose=verbose)
    elif model_name == "resnet34":
        model, preprocess = load_resnet34(verbose=verbose)
    elif model_name == "wide_resnet50_2":
        model, preprocess = load_wideresnet50_2(verbose=verbose)
    elif model_name == "resnext50":
        model, preprocess = load_resnext50(verbose=verbose)
    elif model_name == "mobilenet_v3_large": 
        model, preprocess = load_mobilenet_v3_large(verbose=verbose)
    elif model_name == "densenet121":
        model, preprocess = load_densenet121(verbose=verbose)
    elif model_name == "vit_b_16":
        model, preprocess = load_vit_b_16(verbose=verbose)
    elif model_name == "efficientnet_b0":
        model, preprocess = load_efficientnet_b0(verbose=verbose)
    elif model_name == "efficientnet_b4":
        model, preprocess = load_efficientnet_b4(verbose=verbose)
    elif model_name == "convnext_tiny":
        model, preprocess = load_convnext_tiny(verbose=verbose)
    elif model_name == "swin_t":
        model, preprocess = load_swin_t(verbose=verbose)
    elif model_name == "deit_small":
        model, preprocess = load_deit_small(verbose=verbose)
    else:
        raise ValueError(f"Model {model_name} not recognized. Add it to model_loader.py")
        
    if freeze:
        if verbose:
            print(f"WARNING: Freezing all parameters in {model_name}...")
        for param in model.parameters():
            param.requires_grad = False
    return model, preprocess

def load_efficientnet_b0(verbose=True):
    """
    Loads EfficientNet-B0 pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading EfficientNet-B0 (Pretrained: ImageNet-1K)...")
    weights = models.EfficientNet_B0_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.efficientnet_b0(weights=weights)
    model.eval()
    return model, preprocess

def load_efficientnet_b4(verbose=True):
    """
    Loads EfficientNet-B4 pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading EfficientNet-B4 (Pretrained: ImageNet-1K)...")
    weights = models.EfficientNet_B4_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.efficientnet_b4(weights=weights)
    model.eval()
    return model, preprocess

def load_vit_b_16(verbose=True):
    """
    Loads ViT-B/16 pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading ViT-B/16 (Pretrained: ImageNet-1K)...")
    weights = models.ViT_B_16_Weights.DEFAULT
    preprocess = weights.transforms()

    model = models.vit_b_16(weights=weights)
    model.eval()
    return model, preprocess

def load_densenet121(verbose=True):
    """
    Loads DenseNet-121 pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading DenseNet-121 (Pretrained: ImageNet-1K)...")
    weights=models.DenseNet121_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.densenet121(weights=weights)
    model.eval()
    return model, preprocess

def load_mobilenet_v3_large(verbose=True):
    """
    Loads MobileNetV3-Large pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading MobileNetV3-Large (Pretrained: ImageNet-1K)...")
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.mobilenet_v3_large(weights=weights)
    model.eval()
    return model, preprocess 

def load_resnet50(verbose=True):
    """
    Loads ResNet-50 pretrained on ImageNet-1K. DEFAULT is torchvision's IMAGENET1K_V2 recipe (80.9% top-1, trained
    with label smoothing and mixup, so its outputs are far less confident than the original ResNet-50's); the
    original weights are `resnet50_v1`.
    """
    if verbose:
        print("Loading ResNet-50 (Pretrained: ImageNet-1K)...")
    weights = models.ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    # preprocess = transforms.Compose([
    #     transforms.CenterCrop(224),   # no-op on 224 imgs; no resize, no normalize
    #     transforms.ToTensor(),        # PIL -> [0,1] tensor, that's all
    # ])
    model = models.resnet50(weights=weights)
    model.eval()
    return model, preprocess

def _load_timm_vit_b_16(tag, verbose=True):
    """timm's ViT-B/16 with an explicit pretrained tag. A bare `vit_base_patch16_224` resolves to different weights
    in different timm versions (augreg_in21k_ft_in1k up to 0.6.x, augreg2_in21k_ft_in1k from 0.9), so the tag is
    always spelled out. The preprocessing is the one timm ships with that tag (0.5/0.5 normalization, crop 0.9)."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    name = f"vit_base_patch16_224.{tag}"
    if verbose:
        print(f"Loading ViT-B/16 (timm, {name})...")
    model = timm.create_model(name, pretrained=True)
    model.eval()
    preprocess = create_transform(**resolve_data_config({}, model=model))
    return model, preprocess

def load_vit_b_16_augreg(verbose=True):
    """
    ViT-B/16 AugReg, ImageNet-21k pretrained and fine-tuned on ImageNet-1k (timm `augreg_in21k_ft_in1k`): what a bare
    `vit_base_patch16_224` gave in timm 0.6.x, which the SAR / EATA / COCA code bases were written against. Not
    torchvision's `vit_b_16` (trained on ImageNet-1k only).
    """
    return _load_timm_vit_b_16("augreg_in21k_ft_in1k", verbose)

def load_vit_b_16_augreg2(verbose=True):
    """
    ViT-B/16 AugReg2 (timm `augreg2_in21k_ft_in1k`, a further fine-tune of the AugReg model): what a bare
    `vit_base_patch16_224` gives in timm >= 0.9, including the timm this repo pins.
    """
    return _load_timm_vit_b_16("augreg2_in21k_ft_in1k", verbose)

def load_resnet50_v1(verbose=True):
    """
    Loads ResNet-50 with the ORIGINAL torchvision weights (IMAGENET1K_V1, 76.1% top-1), the ones the Tent paper,
    RobustBench and most TTA benchmarks use. Kept next to `resnet50` (the V2 weights) so that results of the two
    never share a name, a cache or a tuned-settings file.
    """
    if verbose:
        print("Loading ResNet-50 (Pretrained: ImageNet-1K, V1 weights)...")
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    model = models.resnet50(weights=weights)
    model.eval()
    return model, preprocess

def load_resnet18(verbose=True):
    """
    Loads ResNet-18 pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading ResNet-18 (Pretrained: ImageNet-1K)...")
    weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.resnet18(weights=weights)
    model.eval()
    return model, preprocess

def load_resnet34(verbose=True):
    """
    Loads ResNet-34 pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading ResNet-34 (Pretrained: ImageNet-1K)...")
    weights = models.ResNet34_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.resnet34(weights=weights)
    model.eval()
    return model, preprocess

    
def load_convnext_base(verbose=True):
    """
    Loads ConvNeXt-Base pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading ConvNeXt-Base (Pretrained: ImageNet-1K)...")
    weights = models.ConvNeXt_Base_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.convnext_base(weights=weights)
    model.eval()
    return model, preprocess

def load_wideresnet50_2(verbose=True):
    """
    Loads Wide ResNet-50-2 pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading Wide ResNet-50-2 (Pretrained: ImageNet-1K)...")
    weights = models.Wide_ResNet50_2_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.wide_resnet50_2(weights=weights)
    model.eval()
    return model, preprocess

def load_convnext_tiny(verbose=True):
    """
    Loads ConvNeXt-Tiny pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading ConvNeXt-Tiny (Pretrained: ImageNet-1K)...")
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.convnext_tiny(weights=weights)
    model.eval()
    return model, preprocess

def load_swin_t(verbose=True):
    """
    Loads Swin-T pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading Swin-T (Pretrained: ImageNet-1K)...")
    weights = models.Swin_T_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.swin_t(weights=weights)
    model.eval()
    return model, preprocess

def load_deit_small(verbose=True):
    """
    Loads DeiT-S (deit_small_patch16_224, no distillation head) pretrained
    on ImageNet-1K via timm -- torchvision has no DeiT variant.
    """
    if verbose:
        print("Loading DeiT-S (timm, Pretrained: ImageNet-1K)...")
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    model = timm.create_model("deit_small_patch16_224", pretrained=True)
    model.eval()
    preprocess = create_transform(**resolve_data_config({}, model=model))
    return model, preprocess

def load_resnext50(verbose=True):
    """
    Loads ResNeXt-50 pretrained on ImageNet-1K.
    """
    if verbose:
        print("Loading ResNeXt-50 (Pretrained: ImageNet-1K)...")
    weights = models.ResNeXt50_32X4D_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.resnext50_32x4d(weights=weights)
    model.eval()
    return model, preprocess

import torch
def _preprocess_batch(imgs, preprocess, device):
    return torch.stack([preprocess(img) for img in imgs]).to(device)
