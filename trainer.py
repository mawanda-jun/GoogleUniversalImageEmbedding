import yaml
from pathlib import Path
import os
from SimCLR_dataset import GUIE_Dataset, collate_fn, DataLoader, CustomBatchSampler
from model import SimCLRContrastiveLearning

def main(cfg_path: str):
    # Import configuration
    with open(cfg_path, 'r') as file:
        args = yaml.safe_load(file)

    # Redefine args so it takes epochs in consideration
    args['steps'] = args['train_features'] // args['batch_size']
    args['steps'] *= args['epochs']
    args['save_steps'] *= args['epochs']

    # Define model
    model = SimCLRContrastiveLearning(args)

    # Define dataset
    dataset_path = Path(args["dataset_path"])
    train_ids = open(args["train_dataset"], 'r').read().splitlines()
    val_ids = open(args["val_dataset"], 'r').read().splitlines()

    train_set = GUIE_Dataset(
        dataset_path, 
        train_ids, 
        args['multiplier'],
        args['train_features']
        )
    val_set = GUIE_Dataset(
        dataset_path, 
        val_ids, 
        args['multiplier'],
        mock_length=args['val_features']
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=CustomBatchSampler(train_set, args['batch_size']),
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_sampler=CustomBatchSampler(val_set, args['batch_size']),
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    # Transform loaders into iterators - we don't need them anymore!
    train_loader = iter(train_loader)
    val_loader = iter(val_loader)

    # Create model folder
    Path(args["exp_path"]).mkdir(exist_ok=True, parents=True)

    # Save configuration
    with open(Path(args['exp_path']) / Path("config.yaml"), 'w') as writer:
        yaml.safe_dump(data=args, stream=writer)
    
    # Train model
    model.train(train_loader, val_loader)

if "__main__" in __name__:
    cfg_path = "/home/mawanda/projects/GoogleUniversalImageEmbedding/config/config.yaml"
    main(cfg_path)

