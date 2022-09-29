from typing import List
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models.resnet import resnext101_32x8d, ResNeXt101_32X8D_Weights
from torchvision.transforms import ToTensor, Normalize, Resize, Compose


class GRAY2RGB:
    def __call__(self, image: Image):
        image = image.convert("RGB")
        return image


def init_resnet(device):
    model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2).eval().to(device)
    
    starters = list(model.children())[:4]
    bottlenecks = list(model.children())[4:8]
    # Delete ReLU from last bottleneck
    # bottlenecks[-1][-1] = nn.Sequential(*list(bottlenecks[-1][-1].children())[:-1])

    pooler = nn.AdaptiveAvgPool2d((1, 1))

    modules = [*starters, *bottlenecks, pooler]

    extractor = nn.Sequential(*modules)
    for param in extractor.parameters():
        param.requires_grad = False
    
    # Find number of features in output
    num_features = bottlenecks[-1][-1].conv3.out_channels

    # Define preprocess function
    ts = [
        Resize((224, 224)),
        GRAY2RGB(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transform = Compose(ts)
    
    def preprocess(images):
        if not isinstance(images, List):
            images = [images]
        return torch.cat([transform(image) for image in images], 0).to(device)
    
    extractor_fun = lambda images: extractor(images).squeeze(-1).squeeze(-1)

    return preprocess, extractor_fun, num_features