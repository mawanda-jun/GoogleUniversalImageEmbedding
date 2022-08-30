import numpy as np
from typing import List
from utils.count_features_per_category import parse_protobuf_file
from pathlib import Path
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import concurrent.futures

class GUIE_Dataset(Dataset):
    def __init__(
            self, 
            dataset_path: Path, 
            synset_ids: List[str]
        ):
        super().__init__()
        # Memorize dataset information
        self.dataset_path = dataset_path
        self.synset_ids = synset_ids
    
    def __len__(self):
        return len(self.synset_ids)

    def __getitem__(self, idx):
        features, mul = parse_protobuf_file(self.dataset_path / Path(self.synset_ids[idx] + ".pb"))
        features = random.choices(features, k=2)
        features = [f.astype(np.float32) / mul for f in features]
        return features

def collate_fn(batch):
    batch = np.asarray(batch)
    batch_0 = batch[:, 0, ...]
    batch_1 = batch[:, 1, ...]
    return torch.tensor(batch_0), torch.tensor(batch_1)


if "__main__" in __name__:
    dataset = GUIE_Dataset(
        dataset_path = Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding"),
        synset_ids = open("/home/mawanda/projects/GoogleUniversalImageEmbedding/dataset_info/train_synset_ids.txt").read().splitlines(),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1000,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=16,
        pin_memory=True,
        drop_last=True
    )

    for i, data in enumerate(dataloader):
        print(i, data[0].shape, data[1].shape)

        

        

