import os
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
            synset_ids: List[str], 
            batch_size: int,
            limit: int = None
        ):
        super().__init__()
        assert batch_size < len(synset_ids), f"Batch size is too high! Please select any value below {len(synset_ids) - 2}"

        # Memorize dataset information
        self.dataset_path = dataset_path
        self.synset_ids = synset_ids
        self.batch_size = batch_size

        # Load dataset in memory
        self.__load_dataset(limit)

    def __load_dataset(self, limit):
        # Load all features in memory by category
        self.synset_features = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
            synset_features = worker.map(
                parse_protobuf_file, 
                [self.dataset_path / Path(synset_id + ".pb") for synset_id in self.synset_ids]
            )
            for synset_info in tqdm(synset_features, desc="Loading features in memory...", total=len(self.synset_ids)):
                synset_features, mul, synset_path = synset_info
                if limit is not None:
                    features = random.choices(synset_features, k=limit)
                else:
                    features = synset_features
                self.synset_features[str(synset_path.name).split(".")[0]] = features
        self.mul = mul
        self.idx_map = list(self.synset_features.keys())

    def __len__(self):
        return len(self.synset_ids)
    
    # def __getitem__(self, idx):
    #     """This getitem was supposed to extract positive and negative examples.
    #     However I was wrong since this difference is treated inside the criterion.
    #     For now I'm commenting this for future (maybe useful) revision.

    #     Args:
    #         idx (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     # Take negative indexes that are not idx
    #     negative_indexes = set()
    #     while len(negative_indexes) < self.batch_size - 1:  # The positive examples has already been taken!
    #         n_idx = int(round(random.random()*len(self.synset_ids))) 
    #         while n_idx == idx:
    #             n_idx = int(round(random.random()*len(self.synset_ids))) 
    #         negative_indexes.add(n_idx)

    #     positive_features = random.choices(self.synset_features[self.idx_map[idx]], k=2)
    #     negative_features = [
    #         random.choice(self.synset_features[self.idx_map[negative_index]])
    #             for negative_index in negative_indexes
    #     ]
    #     return positive_features, negative_features

    def __getitem__(self, idx):
        feature_0, feature_1 = random.choices(self.synset_features[self.idx_map[idx]], k=2)
        return feature_0, feature_1

    def on_finish(self):
        # Reload different section of dataset
        self.__load_dataset()


def collate_fn(batch):
    batch = batch[0]
    batch[0] = torch.tensor(batch[0])
    batch[1] = torch.tensor(batch[1])
    return batch


if "__main__" in __name__:
    dataset = GUIE_Dataset(
        dataset_path = Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding"),
        synset_ids = open("/home/mawanda/projects/GoogleUniversalImageEmbedding/dataset_info/train_synset_ids.txt").read().splitlines(),
        batch_size = 16,
        limit=1000
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

        

        

