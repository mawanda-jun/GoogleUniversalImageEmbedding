import clip

def init_CLIP(model_type: str = 'ViT-L/14@336px', device: str = 'cuda'):
    model, preprocess = clip.load(model_type, device=device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    def extractor(image): return model.encode_image(image)
    return preprocess, extractor, 768
