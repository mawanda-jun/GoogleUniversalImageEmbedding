import os
from utils.features_pb2 import Features
import numpy as np
from pathlib import Path
from multiprocessing.pool import Pool
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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

    mul = synset_features.multiplier
    features = [np.asarray(representation.features)
             for representation in synset_features.representations]
    
    # image_ids = [
        # representation.image_id for representation in synset_features.representations]

    return features, mul, features_path
    # Modify return for counting reasons
    # return str(features_path.name).split(".")[0], len(features)

def main(dataset_path: Path):
    list_of_pb = list(dataset_path.glob("*.pb"))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        count_per_synsets = worker.map(parse_protobuf_file, list_of_pb)

        txt = open("features_per_class.txt", 'w')
        for count_per_synset in tqdm(count_per_synsets, total=len(list_of_pb)):
            txt.write(f"{count_per_synset[0]} {count_per_synset[1]}\n")
    

if "__main__" in __name__:
    main(Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding"))
    