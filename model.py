from pathlib import Path
import torch
import torch.nn as nn
from modules import ProjectionHead, SimCLR_Loss, LARS
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def load_optimizer(args, model) -> torch.optim.Optimizer:
    if args["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) 
    elif args["optimizer"] == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = args['lr'] * args["batch_size"] / 256
        optimizer = LARS(
            params=model.parameters(),
            lr=learning_rate,
            momentum=args['momentum'],
            weight_decay=args["weight_decay"],
            # exclude_from_weight_decay=["batch_normalization", "bias"],
        )

    else:
        raise NotImplementedError


    # "decay the learning rate with the cosine decay schedule without restarts"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args["epochs"], eta_min=0, last_epoch=-1
    )

    return optimizer, scheduler


class SimCLRContrastiveLearning(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.model = ProjectionHead(
            self.args["in_features"], 
            self.args["hidden_features"],
            self.args["out_features"]
        )
        self.model.to(self.args["device"])

        self.criterion = SimCLR_Loss(self.args["batch_size"], self.args["temperature"])
        self.optimizer, self.scheduler = load_optimizer(args, self.model)

        self.writer = SummaryWriter(log_dir=Path(args["exp_path"]) / Path("log_dir"))

    def save_model(self, epoch):
        out_path = Path(self.args["exp_path"]) / Path(f"checkpoint_{epoch}.tar")
        torch.save(self.model.state_dict(), out_path)

    def train(self, train_loader, val_loader):
        # Put model in train mode
        self.model.train()
        # Train
        for epoch in range(self.args["epochs"]):
            # Find LR
            lr = self.optimizer.param_groups[0]["lr"]

            # Actual training
            train_loader = tqdm(train_loader)
            epoch_loss = 0.
            for (x_i, x_j) in train_loader:
                self.optimizer.zero_grad()
                x_i = x_i.to(self.args["device"], non_blocking=True)
                x_j = x_j.to(self.args["device"], non_blocking=True)
                
                # Positive pair, with encoding
                z_i, z_j = self.model(x_i, x_j)
                loss = self.criterion(z_i, z_j)
                loss.backward()
                self.optimizer.step()

                # Add loss to epoch
                epoch_loss += loss.item()

                # Print some stat
                train_loader.set_description(
                    f"Epoch [{epoch + 1}/{self.args['epochs']}]  Train loss: {loss.item():.4f}  LR: {round(lr, 5)}",
                    refresh=True
                    )
            
            epoch_loss /= len(train_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Put model in eval mode
            self.model.eval()
            eval_loss = 0.
            val_loader = tqdm(val_loader)
            for (x_i, x_j) in val_loader:
                x_i = x_i.to(self.args["device"], non_blocking=True)
                x_j = x_j.to(self.args["device"], non_blocking=True)
                
                # Positive pair, with encoding
                with torch.no_grad():
                    z_i, z_j = self.model(x_i, x_j)
                    loss: torch.Tensor = self.criterion(z_i, z_j)
                # Add loss to epoch
                eval_loss += loss.item()

                # Print some stats
                val_loader.set_description(
                    f"Evaluating: Epoch [{epoch + 1}/{self.args['epochs']}]  Val loss: {loss.item():.4f}  LR: {round(lr, 5)}",
                    refresh=True
                )
            
            eval_loss /= len(val_loader)

            # Save infos to writer
            self.writer.add_scalar("Train/Loss", eval_loss, epoch)
            self.writer.add_scalar("Val/Loss", eval_loss, epoch)
            self.writer.add_scalar("Misc/LR", lr, epoch)
            print(f"Epoch [{epoch + 1}/{self.args['epochs']}]\tTrainLoss: {epoch_loss:.4f}\tValLoss: {eval_loss:.4f}\tlr: {round(lr, 5)}")

            if (epoch+1) % self.args["save_steps"] == 0:
                self.save_model(epoch)




