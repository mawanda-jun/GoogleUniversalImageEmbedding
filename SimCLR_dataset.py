import numpy as np
from typing import List
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import parse_pb


class MultiEpochsDataLoader(DataLoader):
    """
    This dataloader continues prefetching even after epoch is finished. Really useful whenever 
    there are few minibatches and we want to continue training for many many epochs.
    Original is here: https://github.com/rwightman/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py#L209-L238
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class GUIE_Dataset(Dataset):
    def __init__(
            self, 
            dataset_path: Path, 
            synset_ids: List[str],
            multiplier: int
        ):
        super().__init__()
        # Extract information about dataset
        all_files = dataset_path.glob("*.pb")
        self.multiplier = multiplier

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
        return len(self.synset_paths.keys())

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

def collate_fn(batch):
    batch = np.asarray(batch)
    batch_0 = batch[:, 0, ...]
    batch_1 = batch[:, 1, ...]
    return torch.tensor(batch_0), torch.tensor(batch_1)


if "__main__" in __name__:
    dataset = GUIE_Dataset(
        dataset_path = Path("/home/mawanda/Documents/GoogleUniversalImageEmbedding/data/by_chunks"),
        synset_ids = open("/home/mawanda/projects/GoogleUniversalImageEmbedding/dataset_info/train_synset_ids.txt").read().splitlines(),
        multiplier = 10000
    )

    dataloader = MultiEpochsDataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    for i, data in enumerate(dataloader):
        print(i, data[0].shape, data[1].shape)

        

        

