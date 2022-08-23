from threading import Thread
import clip
from pathlib import Path
import wget
from tqdm import tqdm
import tarfile
from PIL import Image
from features_pb2 import Representation, Features
import torch
import numpy as np
import shutil

MULTIPLIER = 10000

def save_features(output_path:Path, features: Representation):
    try:
        with open(output_path, 'wb') as writer:
            writer.write(features.SerializeToString())
        # print(f"Saved representation {str(output_path.name)}!")
    except Exception as e:
        output_path.unlink()
        raise Exception(e)

def parse_protobuf_file(features_path):
    """This method parses a file with a created protobuf.

    Args:
        features_path (str): path/to/features.pb

    Returns:
        (List, List): List of windows information and a representation for each of them.
    """

    # for features_path in feature_paths:
    with open(features_path, 'rb') as data:
        synset_features = Features()
        synset_features.ParseFromString(data.read())
    
    mul = np.array(synset_features.multiplier, dtype=np.float32)
    synset_features = [np.asarray(representation.features, dtype=np.float32) / mul for representation in synset_features.representations]
    image_ids = [representation.image_id for representation in synset_features]

    return synset_features, image_ids

def init_CLIP(model_type: str='ViT-L/14@336px', device:str='cuda'):
    model, preprocess = clip.load(model_type, device=device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    extractor = lambda image: model.encode_image(image)
    return preprocess, extractor

def main(
    batch_size: int,
    output_dir: Path,
    device: str
):
    # Load list of packets to download
    # For each packet:
    # - Download image packet (S)
    # - Extract image packet (S)
    # - Extract features from all images in image packet
    # - tar features packet with same name as the original
    # - delete image packet

    # parse_protobuf_file("/home/mawanda/Documents/GoogleUniversalImageEmbedding/n00004475.pb")
    preprocess, extractor = init_CLIP(device=device)

    synset_ids = open("synset_id.txt", 'r').read().splitlines() 
    # Find synset already downloaded and remove them from list
    for synset in output_dir.glob("*.pb"):
        synset_ids.remove(str(synset.name).split(".")[0] + ".tar")
    
    # Go on!
    for synset_id in tqdm(synset_ids):
        tarpath = output_dir / Path(synset_id)
        if not tarpath.is_file():
            # Download image packet
            url = f"https://image-net.org/data/winter21_whole/{synset_id}"
            wget.download(url, str(output_dir), bar=None)
        
        # Extract image packet
        extraction_output_dir = output_dir / Path(synset_id.split(".")[0])
        extraction_output_dir.mkdir(exist_ok=True, parents=True)
        tar = tarfile.open(tarpath)
        tar.extractall(path=extraction_output_dir)
        tar.close()
        
        # Read all images in folder        
        # Extract features from folder images
        batch = []
        features = []
        for img in extraction_output_dir.glob("*"):
            try:
                img = Image.open(img)
                batch.append(img)
                if len(batch) == batch_size:
                    imgs = [preprocess(img).to(device) for img in batch]
                    features.append(extractor(torch.stack(imgs)).to('cpu').numpy())
                    batch = []
            except Exception:
                continue
        # Process last part of batch
        imgs = [preprocess(img).to(device) for img in batch]
        features.append(extractor(torch.stack(imgs)).to('cpu').numpy())
        features = np.concatenate(features, 0)

        # Load features in proto
        representations = []
        for img_path, features in zip(extraction_output_dir.glob("*"), features):
            representation = Representation()
            representation.features.extend(list((features*MULTIPLIER).astype(np.int32)))
            representation.image_id = str(img_path.name)
            representations.append(representation)
        
        features = Features()
        features.representations.extend(representations)
        features.multiplier = MULTIPLIER

        # Save feature
        output_path = output_dir / Path(synset_id.split(".")[0] + ".pb")
        save_features(output_path=output_path, features=features)

        # Delete old files
        tarpath.unlink()
        shutil.rmtree(extraction_output_dir)
        # Thread(target=save_features, kwargs={"output_path": output_path, "representation": representation}).start()

if "__main__" in __name__:
    batch_size = 128
    output_dir = Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding")
    device='cuda'
    main(batch_size, output_dir, device)
