from peewee import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from collections import OrderedDict
import data_parser

global model


class EvaluationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=1024, layer_count=6):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        layers = []
        
        # Model V1
        # layers.append(("linear-0", nn.Linear(896, 896)))
        # layers.append(("relu-0", nn.ReLU()))
        # layers.append(("linear-1", nn.Linear(896, 448)))
        # layers.append(("relu-1", nn.ReLU()))
        # layers.append(("linear-2", nn.Linear(448, 224)))
        # layers.append(("relu-2", nn.ReLU()))
        # layers.append(("linear-3", nn.Linear(224, 112)))
        # layers.append(("relu-3", nn.ReLU()))
        # layers.append(("linear-4", nn.Linear(112, 1)))
        
        # Model V2
        # for every layer decrease the size by half, keep track of previous size
        # prev_size = 896
        # for i in range(1, layer_count):
        #     layers.append((f"linear-{i}", nn.Linear(prev_size, prev_size // 2)))
        #     layers.append((f"relu-{i}", nn.ReLU()))
        #     prev_size = prev_size // 2
        # layers.append((f"linear-{layer_count}", nn.Linear(prev_size, 1)))

        # Model V3
        # layers.append(("linear-0", nn.Linear(896, 2048)))
        # layers.append(("relu-0", nn.ReLU()))
        # for i in range(1, 6):
        #     layers.append((f"linear-{i}", nn.Linear(2048, 2048)))
        #     layers.append((f"relu-{i}", nn.ReLU()))

        # layers.append(("linear-6", nn.Linear(2048, 1)))
        
        # Model V3 with dropout and batch normalization
        layers.append(("linear-0", nn.Linear(896, 2048)))
        layers.append(("batchnorm-0", nn.BatchNorm1d(2048)))  # Add batch normalization
        layers.append(("relu-0", nn.ReLU()))
        layers.append(("dropout-0", nn.Dropout(0.5)))  # Add dropout
        for i in range(1, 6):
            layers.append((f"linear-{i}", nn.Linear(2048, 2048)))
            layers.append((f"batchnorm-{i}", nn.BatchNorm1d(2048)))  # Add batch normalization
            layers.append((f"relu-{i}", nn.ReLU()))
            layers.append((f"dropout-{i}", nn.Dropout(0.5)))  # Add dropout

        layers.append(("linear-6", nn.Linear(2048, 1)))
        
        self.seq = nn.Sequential(OrderedDict(layers))
        
    def forward(self, x):
        x = x.view(-1, 896)
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.l1_loss(y_hat, y.squeeze())
        print("loss", loss)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = data_parser.EvaluationDataset()
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True
        )


if __name__ == "__main__":
    configs = [
        {"layer_count": 6, "batch_size": 1024},
    ]
    for config in configs:
        version_name = (
            f'MBDbatch_size-{config["batch_size"]}-layer_count-{config["layer_count"]}'
        )
        logger = pl.loggers.TensorBoardLogger(
            "lightning_logs", name="chessml", version=version_name
        )
        trainer = pl.Trainer(precision=16, logger=logger, max_epochs=10)
        model = EvaluationModel(
            batch_size=config["batch_size"],
            learning_rate=1e-3,
            layer_count=config["layer_count"],
        )
        trainer.tune(model)
        lr_finder = trainer.tuner.lr_find(model, min_lr=1e-6, max_lr=1e-3, num_training=25)
        trainer.fit(model)

        trainer.save_checkpoint(f"checkpoints/{version_name}.ckpt")
