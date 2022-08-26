from time import sleep
import clip
from pathlib import Path
from tqdm import tqdm
import tarfile
from PIL import Image
from features_pb2 import Representation, Features
import torch
import numpy as np
import shutil
from multiprocessing.pool import Pool

MULTIPLIER = 10000


def save_features(output_path: Path, features: Representation):
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
    features = [np.asarray(representation.features, dtype=np.float32) /
                mul for representation in synset_features.representations]
    image_ids = [
        representation.image_id for representation in synset_features.representations]

    return features, image_ids


def init_CLIP(model_type: str = 'ViT-L/14@336px', device: str = 'cuda'):
    model, preprocess = clip.load(model_type, device=device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    def extractor(image): return model.encode_image(image)
    return preprocess, extractor


def extract_tar(args):
    tarpath, member, output_dir = args
    tar = tarfile.open(tarpath)
    tar.extract(member, output_dir)


def extract_save(
    preprocess,
    extractor,
    output_dir: Path,
    batch_size,
    device
):
    tarpaths = list(output_dir.glob("*.tar"))
    for tarpath in tqdm(tarpaths):
        # Extract image packet
        extraction_output_dir = output_dir / \
            Path(str(tarpath.name).split(".")[0])
        extraction_output_dir.mkdir(exist_ok=True, parents=True)
        # Enable multiprocessing extraction
        tar = tarfile.open(tarpath)
        # tarmembers = tar.getmembers()
        # pool = Pool(processes=os.cpu_count())
        # pool.map(
        #     extract_tar,
        #     [(tarpath, tarmember, extraction_output_dir) for tarmember in tarmembers]
        # )
        # # for tarf, tarmember, od in zip(repeat(tar, len(tarmembers)), tarmembers, repeat(extraction_output_dir, len(tarmembers))):
        # #     extract_tar(tarf, tarmember, od)
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
                    features.append(
                        extractor(torch.stack(imgs)).to('cpu').numpy())
                    batch = []
            except Exception as e:
                print(e)
                continue
        # Process last part of batch
        if len(batch) > 0:
            imgs = [preprocess(img).to(device) for img in batch]
            features.append(extractor(torch.stack(imgs)).to('cpu').numpy())

        if len(imgs) == 0 and len(features) == 0:
            print(
                f"Something went wrong with {tarpath}, so we are skipping it for now...")
            tarpath.unlink()
            continue

        features = np.concatenate(features, 0)

        # Load features in proto
        representations = []
        for img_path, features in zip(extraction_output_dir.glob("*"), features):
            representation = Representation()
            representation.features.extend(
                list((features*MULTIPLIER).astype(np.int32)))
            representation.image_id = str(img_path.name)
            representations.append(representation)

        features = Features()
        features.representations.extend(representations)
        features.multiplier = MULTIPLIER

        # Save feature
        output_path = output_dir / \
            Path(str(tarpath.name).split(".")[0] + ".pb")
        save_features(output_path=output_path, features=features)

        # Delete old files
        tarpath.unlink()
        shutil.rmtree(extraction_output_dir)
        # Thread(target=save_features, kwargs={"output_path": output_path, "representation": representation}).start()


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

    # parse_protobuf_file("/home/mawanda/Documents/GoogleUniversalImageEmbedding/n00015388.pb")
    preprocess, extractor = init_CLIP(device=device)
    while True:
        _ = extract_save(preprocess, extractor, output_dir, batch_size, device)
        print("Now sleeping 1min waiting for new tar...")
        sleep(60)


if "__main__" in __name__:
    batch_size = 256
    output_dir = Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding")
    device = 'cuda'
    main(batch_size, output_dir, device)
