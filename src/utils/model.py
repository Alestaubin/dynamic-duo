import torchvision.models as models

def get_model(model_name, freeze=True):
    """
    Dispatcher function to load models by name.
    """
    model_name = model_name.lower().strip()
    
    if model_name == "resnet50":
        model, preprocess = load_resnet50()
    elif model_name == "convnext_base":
        model, preprocess = load_convnext_base()
    elif model_name == "resnet18":
        model, preprocess = load_resnet18()
    elif model_name == "resnet34":
        model, preprocess = load_resnet34()
    elif model_name == "wide_resnet50_2":
        model, preprocess = load_wideresnet50_2()
    elif model_name == "resnext50":
        model, preprocess = load_resnext50()
    elif model_name == "mobilenet_v3_large": 
        model, preprocess = load_mobilenet_v3_large()
    elif model_name == "densenet121":
        model, preprocess = load_densenet121()
    elif model_name == "vit_b_16":
        model, preprocess = load_vit_b_16()
    elif model_name == "efficientnet_b0":
        model, preprocess = load_efficientnet_b0()
    else:
        raise ValueError(f"Model {model_name} not recognized. Add it to model_loader.py")
        
    if freeze:
        print(f"WARNING: Freezing all parameters in {model_name}...")
        for param in model.parameters():
            param.requires_grad = False
    return model, preprocess

def load_efficientnet_b0():
    """
    Loads EfficientNet-B0 pretrained on ImageNet-1K.
    """
    print("Loading EfficientNet-B0 (Pretrained: ImageNet-1K)...")
    weights = models.EfficientNet_B0_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.efficientnet_b0(weights=weights)
    model.eval()
    return model, preprocess

def load_vit_b_16():
    """
    Loads ViT-B/16 pretrained on ImageNet-1K.
    """
    print("Loading ViT-B/16 (Pretrained: ImageNet-1K)...")
    weights = models.ViT_B_16_Weights.DEFAULT
    preprocess = weights.transforms()

    model = models.vit_b_16(weights=weights)
    model.eval()
    return model, preprocess

def load_densenet121():
    """
    Loads DenseNet-121 pretrained on ImageNet-1K.
    """
    print("Loading DenseNet-121 (Pretrained: ImageNet-1K)...")
    weights=models.DenseNet121_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.densenet121(weights=weights)
    model.eval()
    return model, preprocess

def load_mobilenet_v3_large():
    """
    Loads MobileNetV3-Large pretrained on ImageNet-1K.
    """
    print("Loading MobileNetV3-Large (Pretrained: ImageNet-1K)...")
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.mobilenet_v3_large(weights=weights)
    model.eval()
    return model, preprocess 

def load_resnet50():
    """
    Loads ResNet-50 pretrained on ImageNet-1K.
    """
    print("Loading ResNet-50 (Pretrained: ImageNet-1K)...")
    weights = models.ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.resnet50(weights=weights)
    model.eval() 
    return model, preprocess

def load_resnet18():
    """
    Loads ResNet-18 pretrained on ImageNet-1K.
    """
    print("Loading ResNet-18 (Pretrained: ImageNet-1K)...")
    weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.resnet18(weights=weights)
    model.eval()
    return model, preprocess

def load_resnet34():
    """
    Loads ResNet-34 pretrained on ImageNet-1K.
    """
    print("Loading ResNet-34 (Pretrained: ImageNet-1K)...")
    weights = models.ResNet34_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.resnet34(weights=weights)
    model.eval()
    return model, preprocess

    
def load_convnext_base():
    """
    Loads ConvNeXt-Base pretrained on ImageNet-1K.
    """
    print("Loading ConvNeXt-Base (Pretrained: ImageNet-1K)...")
    weights = models.ConvNeXt_Base_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.convnext_base(weights=weights)
    model.eval()
    return model, preprocess

def load_wideresnet50_2():
    """
    Loads Wide ResNet-50-2 pretrained on ImageNet-1K.
    """
    print("Loading Wide ResNet-50-2 (Pretrained: ImageNet-1K)...")
    weights = models.Wide_ResNet50_2_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.wide_resnet50_2(weights=weights)
    model.eval()
    return model, preprocess

def load_resnext50():
    """
    Loads ResNeXt-50 pretrained on ImageNet-1K.
    """
    print("Loading ResNeXt-50 (Pretrained: ImageNet-1K)...")
    weights = models.ResNeXt50_32X4D_Weights.DEFAULT
    preprocess = weights.transforms()
    model = models.resnext50_32x4d(weights=weights)
    model.eval()
    return model, preprocess
