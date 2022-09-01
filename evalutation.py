import os
from clip.clip import _download, _MODELS
import torch
import torch.nn as nn

from modules.projection_head import ProjectionHead
import torchvision.transforms.functional as F

def init_CLIP(device):
    clip_code = 'ViT-L/14@336px'
    model_path = _download(_MODELS[clip_code], os.path.expanduser(".cache"))
    with open(model_path, 'rb') as opened_file:
        clip_vit = torch.jit.load(opened_file, map_location=device).visual.eval()
    return clip_vit

class CustomProjectionHead(ProjectionHead):
    def __init__(self, device):
        super().__init__(in_features=768, hidden_features=1024, out_features=64)
        # Load weights
        super().load_state_dict(torch.load("/home/mawanda/Documents/GoogleUniversalImageEmbedding/experiments/first_trial/checkpoint_10599.tar", map_location=device))
    
    def forward(self, x):
        return self.projection(x)


class MarmittoniModel(nn.Module):
  def __init__(self, device):
    super().__init__()
    self.extractor = init_CLIP(device)
    self.mlp_head = CustomProjectionHead(device).eval()

  def forward(self, x):
    x = F.resize(x, size=(336, 336), interpolation=F.InterpolationMode.BICUBIC)
    x = F.center_crop(x, output_size=(336, 336))
    x /= 255.
    x = F.normalize(x, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    x = self.extractor(x)
    with torch.no_grad():
        x = self.mlp_head(x)
    return x

def create_model():
    model = MarmittoniModel('cpu')
    model.eval()
    saved_model = torch.jit.script(model)
    saved_model.save('saved_model.pt')

if "__main__" in __name__:
    create_model()