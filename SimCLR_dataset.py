import numpy as np
from typing import List
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from utils import parse_pb


class GUIE_Dataset(Dataset):
    def __init__(
            self, 
            dataset_path: Path, 
            synset_ids: List[str],
            multiplier: int,
            mock_length: int
        ):
        """_summary_
        Use mock_length so dataloader is resetted only after one epoch - and not after
        finishing the far-less classes.
        Args:
            dataset_path (Path): _description_
            synset_ids (List[str]): _description_
            multiplier (int): _description_
            mock_length (int): _description_
        """
        super().__init__()
        # Extract information about dataset
        all_files = dataset_path.glob("*.pb")
        self.multiplier = multiplier
        self.mock_length = mock_length
        self.synset_ids = synset_ids

        # Index file by category: keep path/to/cat/chunks together, indexed by synset cat id provided.
        self.synset_paths = {k: [] for k in synset_ids}
        for file in tqdm(all_files, desc='Creating index...', total=6581546):
            cat = str(file.name).split("_")[0]
            try:
                self.synset_paths[cat].append(file)
            except KeyError:
                # File is not present in this dataset portion, so skip it!
                continue

        # Create mapping
        self.idx_mapper = [synset_id for synset_id in self.synset_paths.keys()]
    
    def __len__(self):
        return self.mock_length

    def __getitem__(self, idx):
        # Rationale of getitem:
        # - idx select which category we are dealing with
        # - from the category, select two random paths: we will extract the features from these two files
        #   -> the two paths might be the same (random.choices instead of random.sample). This is 
        #      intentional since we want to possibly return the same features, or the features inside the
        #      same file
        # - parse the two paths and extract the features. For each file, in this configuration, there will be
        #   2 features.
        # - select one of the two features for each file.
        # Now we have two features of the same category, which we are going to return.
        synset_paths = self.synset_paths[self.idx_mapper[idx]]
        selected_synset_paths = random.choices(population=synset_paths, k=2)
        
        features = [random.sample(parse_pb(path)[0], 1)[0] for path in selected_synset_paths]
        features = [f.astype(np.float32) / self.multiplier for f in features]
        return features
        # return idx

def collate_fn(batch):
    batch = np.asarray(batch)
    batch_0 = batch[:, 0, ...]
    batch_1 = batch[:, 1, ...]
    return torch.tensor(batch_0), torch.tensor(batch_1)


class CustomBatchSampler(Sampler):
    """
    Create custom batch sampler in order to create a batch of indexes that
    - are not repeated

    Args:
        Sampler (_type_): _description_
    """
    def __init__(self, dataset: GUIE_Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.real_index = list(range(len(self.dataset.synset_ids)))
        assert len(self.real_index) >= batch_size, f"Not enough elements for a batch! Please input a batch size <= {len(self.real_index)}"

    def generate_batch(self):
        while True:
            yield random.sample(self.real_index, self.batch_size)

    def __iter__(self):
        return iter(self.generate_batch())

    def __len__(self):
        return len(self.dataset)

if "__main__" in __name__:
    dataset = GUIE_Dataset(
        dataset_path = Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding/data/by_chunks"),
        synset_ids = open("/home/mawanda/projects/GoogleUniversalImageEmbedding/dataset_info/train_synset_ids.txt").read().splitlines(),
        multiplier = 10000,
        epoch_mul=100
    )
    batch_size = 15332
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=CustomBatchSampler(dataset, batch_size),
        # collate_fn=collate_fn,
        num_workers=10,
        pin_memory=True
    )

    for i, data in enumerate(dataloader):
        print(i, data[0].shape, data[1].shape)

        

        

