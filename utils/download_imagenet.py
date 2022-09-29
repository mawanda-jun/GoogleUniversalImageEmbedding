from itertools import repeat
import os
import wget
from tqdm import tqdm
import concurrent.futures
from pathlib import Path

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


def main(synset_ids, output_dir: Path):
    # Find synset already downloaded and remove them from list
    for synset in output_dir.glob("*.pb"):
        synset_ids.remove(str(synset.name).split(".")[0])

    # Go on!
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        list(tqdm(worker.map(download_and_save,
            synset_ids,
            repeat(output_dir, len(synset_ids)),
        ), total=len(synset_ids)))


if "__main__" in __name__:
    synset_ids = open("/projects/GoogleUniversalImageEmbedding/dataset_info/good_synset_ids.txt", 'r').read().splitlines()
    output_dir = Path("/data/GoogleUniversalImageEmbedding/data/ResNeXt101_32X8D/by_cat")
    output_dir.mkdir(exist_ok=True, parents=True)
    main(synset_ids, output_dir)
