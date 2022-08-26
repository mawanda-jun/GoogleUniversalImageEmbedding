from itertools import repeat
import os
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
import concurrent.futures


def download_and_save(synset_id: str, output_dir: Path):
    tarpath = output_dir / Path(synset_id + ".tar")
    if not tarpath.is_file():
        # Download image packet
        url = f"https://image-net.org/data/winter21_whole/{synset_id}.tar"
        try:
            wget.download(url, str(output_dir), bar=None)
        except Exception as e:
            print(e)
            print(f"Problems with {url}, deleting file if existing...")
            if tarpath.is_file:
                tarpath.unlink()
    else:
        print(f"Skipping {tarpath}...")
        return


def main(output_dir: Path):
    synset_ids = open("synset_ids.txt", 'r').read().splitlines()
    # Find synset already downloaded and remove them from list
    for synset in output_dir.glob("*.pb"):
        synset_ids.remove(str(synset.name).split(".")[0])

    # Go on!
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        list(tqdm(worker.map(download_and_save,
            synset_ids,
            repeat(output_dir, len(synset_ids)),
        ), total=len(synset_ids)))

    # for synset_id in tqdm(synset_ids):
    #     download_extract_save(
    #         synset_id,
    #         output_dir,
    #         preprocess,
    #         extractor,
    #         batch_size,
    #         device)


if "__main__" in __name__:
    output_dir = Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding")
    main(output_dir)
