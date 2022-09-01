from pathlib import Path
import torch
import torch.nn as nn
from modules import ProjectionHead, SimCLR_Loss, init_optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter


def load_optimizer(args, model) -> torch.optim.Optimizer:
    # optimized using LARS with linear learning rate scaling
    # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
    optimizer = init_optim(model=model, args=args)

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
    #     # Make loaders iterable
    #     train_loader = iter(train_loader)
    #     val_loader = iter(val_loader)
        # Initialize gradscaler
        scaler = GradScaler()

        # Put model in train mode
        self.model.train()
        # Train
        epochs = tqdm(range(self.args['epochs']))
        for epoch in epochs:
            # Find LR
            lr = self.optimizer.param_groups[0]["lr"]

            # Actual training
            epoch_loss = 0.
            # for train_step in range(self.args['train_minibatches']):
            for (x_i, x_j) in train_loader:
                # x_i, x_j = next(train_loader)
                self.optimizer.zero_grad()
                x_i = x_i.to(self.args["device"], non_blocking=True)
                x_j = x_j.to(self.args["device"], non_blocking=True)
                
                with autocast():
                    # Positive pair, with encoding
                    z_i, z_j = self.model(x_i, x_j)
                    loss = self.criterion(z_i, z_j)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Add loss to epoch
                epoch_loss += loss.item()
            
            # epoch_loss /= len(self.args['train_minibatches'])
            epoch_loss /= len(train_loader)

            # Print some stat
            epochs.set_description(
                f"Epoch [{epoch + 1}/{self.args['epochs']}]  TrainLoss: {epoch_loss:.4f}  LR: {lr:.6f}",
                refresh=False
                )

            # Save infos to writer
            self.writer.add_scalar("Train/Loss", epoch_loss, epoch)
            self.writer.add_scalar("Misc/LR", lr, epoch)

            # Update scheduler
            self.scheduler.step()

            # Evaluate if it's time
            if (epoch+1) % self.args["val_steps"] == 0:
                # Put model in eval mode
                self.model.eval()
                eval_loss = 0.
                # for val_step in range(self.args['val_minibatches']):
                #   x_i, x_j = next(val_loader)
                for (x_i, x_j) in val_loader:
                    x_i = x_i.to(self.args["device"], non_blocking=True)
                    x_j = x_j.to(self.args["device"], non_blocking=True)
                    
                    # Positive pair, with encoding
                    with torch.no_grad():
                        z_i, z_j = self.model(x_i, x_j)
                        loss: torch.Tensor = self.criterion(z_i, z_j)
                    # Add loss to epoch
                    eval_loss += loss.item()
                
                eval_loss /= len(val_loader)
                
                # Print some stats
                epochs.set_description(
                    f"Evaluating: Epoch [{epoch + 1}/{self.args['epochs']}]  ValLoss: {eval_loss:.4f}  LR: {lr:.6f}",
                    refresh=False
                )
                self.writer.add_scalar("Val/Loss", eval_loss, epoch)

            if (epoch+1) % self.args["save_steps"] == 0:
                self.save_model(epoch)
            
            # Flush writer
            self.writer.flush()




