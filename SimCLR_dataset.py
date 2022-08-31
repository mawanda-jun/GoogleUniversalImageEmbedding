import numpy as np
from typing import List
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.protobuf_utils import parse_pb

class GUIE_Dataset(Dataset):
    def __init__(
            self, 
            dataset_path: Path, 
            synset_ids: List[str]
        ):
        super().__init__()
        # Extract information about dataset
        all_files = dataset_path.glob("*.pb")

        # Index file by category
        self.synset_paths = {k: [] for k in synset_ids}
        for file in tqdm(all_files, desc='Creating index...', total=1324214):
            cat = str(file.name).split("_")[0]
            try:
                self.synset_paths[cat].append(file)
            except KeyError:
                # File is not present in this dataset, so skip it!
                continue

        # Create mapping
        self.idx_mapper = [synset_id for synset_id in self.synset_paths.keys()]
    
    def __len__(self):
        return len(self.synset_paths.keys())

    def __getitem__(self, idx):
        synset_paths = self.synset_paths[self.idx_mapper[idx]]
        selected_synset_paths = random.choices(synset_paths, k=2)
        
        if len(selected_synset_paths) == 1:
            keep_features = random.choices(features, k=2)
        else:
            features = []
            for path in selected_synset_paths:
                f, image_ids, mul = parse_pb(path)
                features.extend(f)
            keep_features = random.choices(features, k=2)
        
        # features = random.choices(features, k=2)
        features = [f.astype(np.float32) / mul for f in keep_features]
        return features

def collate_fn(batch):
    batch = np.asarray(batch)
    batch_0 = batch[:, 0, ...]
    batch_1 = batch[:, 1, ...]
    return torch.tensor(batch_0), torch.tensor(batch_1)


if "__main__" in __name__:
    dataset = GUIE_Dataset(
        dataset_path = Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding/single_features"),
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

        

        

