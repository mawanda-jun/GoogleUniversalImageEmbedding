from time import sleep
from CLIP_utils import init_CLIP
from resnet_utils import init_resnet
from pathlib import Path
from tqdm import tqdm
import tarfile
from PIL import Image
import torch
import numpy as np
import shutil
from protobuf_utils import create_pb, save_features

MULTIPLIER = 10000


def extract_tar(args):
    tarpath, member, output_dir = args
    tar = tarfile.open(tarpath)
    tar.extract(member, output_dir)


def extract_save(
    models,
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
        for model_name, (preprocess, extractor) in models.items():
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
            paths = extraction_output_dir.glob("*")
            # Load features in proto
            features = create_pb(features, paths, MULTIPLIER)
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
    clip_preprocess, clip_extractor, clip_features = init_CLIP(device=device)
    resnet_preprocess, resnet_extractor, resnet_features = init_resnet(device=device)
    
    models = {
        # 'CLIP': (clip_preprocess, clip_extractor),
        'ResNeXt101_32X8D': (resnet_preprocess, resnet_extractor)
        }

    while True:
        _ = extract_save(models, output_dir, batch_size, device)
        print("Now sleeping 1min waiting for new tar...")
        sleep(10)


if "__main__" in __name__:
    batch_size = 256
    output_dir = Path("/data/GoogleUniversalImageEmbedding/data/ResNeXt101_32X8D/by_cat")
    device = 'cuda'
    main(batch_size, output_dir, device)
