from .protobuf_utils import parse_pb, create_pb, save_features
import concurrent.futures
import os
from tqdm import tqdm
from itertools import repeat
from pathlib import Path

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_file(features_file: str, chunk_size: int, out_dir: Path):
    features, image_ids, mul = parse_pb(features_file)
    for i, (c_features, c_image_id) in enumerate(zip(chunks(features, chunk_size), chunks(image_ids, chunk_size))):
        out_file = out_dir / Path(str(features_file.name).split(".")[0] + f"_{i}.pb")
        features = create_pb(c_features, c_image_id, 1)
        save_features(out_file, features)

def main(
    features_path: Path,
    chunked_features_path: Path,
    chunk_size: int
):
    # Gather all features files
    features_files = list(features_path.glob("*.pb"))
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        _ = list(tqdm(worker.map(
                process_file,
                features_files,
                repeat(chunk_size),
                repeat(chunked_features_path)
        ), total=len(features_files)))

def unlink(path: Path): path.unlink()

def delete_dataset(base_path: Path):
    for path in tqdm(base_path.glob("*.pb"), total=6581546):
        path.unlink()
    # with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
    #     print("Loading paths...")
    #     paths = list(base_path.glob("*.pb"))
    #     _ = list(tqdm(worker.map(unlink, paths), total=len(paths)))        

if "__main__" in __name__:
    base_path = Path('/data/GoogleUniversalImageEmbedding/data/CLIP')
    features_path = base_path / Path("by_cat")
    chunked_features_path = base_path / Path("by_chunks")
    chunked_features_path.mkdir(exist_ok=True, parents=True)
    chunk_size = 2
    print("Deleting old dataset...")
    delete_dataset(chunked_features_path)
    print("Making new dataset...")
    main(features_path, chunked_features_path, chunk_size)
