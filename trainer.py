import yaml
from pathlib import Path
from torch.utils.data import DataLoader
import os
from SimCLR_dataset import GUIE_Dataset, collate_fn
from model import SimCLRContrastiveLearning

def main(cfg_path: str):
    # Import configuration
    with open(cfg_path, 'r') as file:
        args = yaml.safe_load(file)

    # Define dataset
    dataset_path = Path(args["dataset_path"])
    train_ids = open(args["train_dataset"], 'r').read().splitlines()
    val_ids = open(args["val_dataset"], 'r').read().splitlines()
    
    train_set = GUIE_Dataset(dataset_path, train_ids)
    val_set = GUIE_Dataset(dataset_path, val_ids)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    # Define model
    model = SimCLRContrastiveLearning(args)

    # Train model
    Path(args["exp_path"]).mkdir(exist_ok=True, parents=True)
    model.train(train_loader, val_loader)

if "__main__" in __name__:
    cfg_path = "/home/mawanda/projects/GoogleUniversalImageEmbedding/config/config.yaml"
    main(cfg_path)

