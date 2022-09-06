import os
import numpy as np
from pathlib import Path
from multiprocessing.pool import Pool
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def main(dataset_path: Path):
    list_of_pb = list(dataset_path.glob("*.pb"))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        count_per_synsets = worker.map(parse_protobuf_file, list_of_pb)

        txt = open("features_per_class.txt", 'w')
        for count_per_synset in tqdm(count_per_synsets, total=len(list_of_pb)):
            txt.write(f"{count_per_synset[0]} {count_per_synset[1]}\n")
    

if "__main__" in __name__:
    main(Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding"))
    