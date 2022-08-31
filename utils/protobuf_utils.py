from pathlib import Path
from typing import List
from utils.features_pb2 import Representation, Features
import numpy as np

def create_pb(features: List[np.ndarray], paths: List[Path], multiplier):
    # Load features in proto
    representations = []
    for img_path, features in zip(paths, features):
        representation = Representation()
        representation.features.extend(
            list((features*multiplier).astype(np.int32)))
        representation.image_id = img_path
        representations.append(representation)

    features = Features()
    features.representations.extend(representations)
    features.multiplier = multiplier

    return features


def parse_pb(features_path):
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
    
    image_ids = [
        representation.image_id for representation in synset_features.representations]

    return features, image_ids, mul

def save_features(output_path: Path, features: Features):
    try:
        with open(output_path, 'wb') as writer:
            writer.write(features.SerializeToString())
        # print(f"Saved representation {str(output_path.name)}!")
    except Exception as e:
        output_path.unlink()
        raise Exception(e)
