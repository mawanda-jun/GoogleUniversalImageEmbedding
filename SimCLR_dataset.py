import os
from typing import List
from utils.count_features_per_category import parse_protobuf_file
from pathlib import Path
from tqdm import tqdm
import random

from torch.utils.data import Dataset, DataLoader
import yaml
import concurrent.futures

class GUIE_Dataset(Dataset):
    def __init__(
            self, 
            dataset_path: Path, 
            synset_ids: List[str], 
            batch_size: int
        ):
        super().__init__()
        assert batch_size < len(synset_ids), f"Batch size is too high! Please select any value below {len(synset_ids) - 2}"
        # Load all features in memory by category
        self.synset_ids = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
            synset_features = worker.map(
                parse_protobuf_file, 
                [dataset_path / Path(synset_id + ".pb") for synset_id in synset_ids]
                )
            for synset_info in tqdm(synset_features, desc="Loading features in memory...", total=len(synset_ids)):
                synset_features, mul, synset_path = synset_info
                self.synset_ids[str(synset_path.name).split(".")[0]] = random.choices(synset_features, k=1000)
        self.mul = mul
        self.batch_size = batch_size
        self.idx_map = list(self.synset_ids.keys())

    def __len__(self):
        return len(self.synset_ids)
    
    def __getitem__(self, idx):
        batch_size = self.batch_size - 1  # The positive examples has already been taken!
        # Take negative indexes from below idx
        bottom_index = max(0, idx - batch_size)
        negative_indexes = list(range(bottom_index, idx))
        if idx - bottom_index < batch_size:  # idx < batch_size
            negative_indexes.extend(list(range(idx+1, batch_size - idx)))

        positive_features = random.choices(self.synset_ids[self.idx_map[idx]], k=2)
        negative_features = [
            random.choice(self.synset_ids[self.idx_map[negative_index]])
                for negative_index in negative_indexes
        ]
        return positive_features, negative_features


def collate_fn(batch):
    return batch[0]


if "__main__" in __name__:
    dataset = GUIE_Dataset(
        dataset_path = Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding"),
        synset_ids = open("/home/mawanda/projects/GoogleUniversalImageEmbedding/dataset_info/train_synset_ids.txt").read().splitlines(),
        batch_size = 4096
    )

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=True
    )

    for data in dataloader:
        print(data)
        break

        

        

