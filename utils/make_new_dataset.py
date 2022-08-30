from features_pb2 import Representation
from count_features_per_category import parse_protobuf_file
from compress_imagenet import save_features
import concurrent.futures
import os
from tqdm import tqdm
from itertools import repeat
from pathlib import Path


def process_file(features_file: str, out_dir: Path):
    representations, image_ids, mul = parse_protobuf_file(features_file)
    for feature, image_id in zip(representations, image_ids):
        representation = Representation()
        representation.features.extend(
            list(feature))
        representation.image_id = image_id
        out_file = out_dir / Path(str(features_file.name).split(".")[0] + "__" + image_id.split(".")[0] + ".pb")
        save_features(out_file, representation)

def main():
    base_path = Path('/home/mawanda/Documents/GoogleUniversalImageEmbedding')
    features_path = base_path / Path("data")
    new_features_path = base_path / Path("single_features")
    new_features_path.mkdir(exist_ok=True, parents=True)

    # Gather all features files
    features_files = list(features_path.glob("*.pb"))
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        _ = list(tqdm(worker.map(
                process_file,
                features_files,
                repeat(new_features_path)
        ), total=len(features_files)))
        

if "__main__" in __name__:
    main()
