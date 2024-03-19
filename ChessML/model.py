from peewee import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from collections import OrderedDict
import data_parser

global model


class EvaluationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=1024, layer_count=6):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        layers = []

        # Model V3
        # layers.append(("linear-0", nn.Linear(896, 2048)))
        # layers.append(("relu-0", nn.ReLU()))
        # for i in range(1, 6):
        #     layers.append((f"linear-{i}", nn.Linear(2048, 2048)))
        #     layers.append((f"relu-{i}", nn.ReLU()))

        # layers.append(("linear-6", nn.Linear(2048, 1)))

        # Model V3 with dropout and batch normalization
        # layers.append(("linear-0", nn.Linear(896, 2048)))
        # layers.append(("batchnorm-0", nn.BatchNorm1d(2048)))  # Add batch normalization
        # layers.append(("relu-0", nn.ReLU()))
        # layers.append(("dropout-0", nn.Dropout(0.5)))  # Add dropout
        # for i in range(1, 6):
        #     layers.append((f"linear-{i}", nn.Linear(2048, 2048)))
        #     layers.append((f"batchnorm-{i}", nn.BatchNorm1d(2048)))  # Add batch normalization
        #     layers.append((f"relu-{i}", nn.ReLU()))
        #     layers.append((f"dropout-{i}", nn.Dropout(0.5)))  # Add dropout

        # layers.append(("linear-6", nn.Linear(2048, 1)))

        # Model V4 with leaky relu and more layers
        layers.append(("linear-0", nn.Linear(896, 4096)))
        layers.append(("batchnorm-0", nn.BatchNorm1d(4096)))  # Add batch normalization
        layers.append(("leakyrelu-0", nn.LeakyReLU()))
        layers.append(("dropout-0", nn.Dropout(0.5)))  # Add dropout
        layers.append(("linear-1", nn.Linear(4096, 4096)))
        layers.append(("batchnorm-1", nn.BatchNorm1d(4096)))  # Add batch normalization
        layers.append(("leakyrelu-1", nn.LeakyReLU()))
        layers.append(("dropout-1", nn.Dropout(0.5)))  # Add dropout
        prev = 4096
        for i in range(2, 6):
            layers.append((f"linear-{i}", nn.Linear(prev, prev // 2)))
            layers.append(
                (f"batchnorm-{i}", nn.BatchNorm1d(prev // 2))
            )  # Add batch normalization
            layers.append((f"leakyrelu-{i}", nn.LeakyReLU()))
            layers.append((f"dropout-{i}", nn.Dropout(0.5)))  # Add dropout
            prev = prev // 2

        layers.append(("linear-6", nn.Linear(prev, 1)))

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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
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
            f'M6batch_size-{config["batch_size"]}-layer_count-{config["layer_count"]}'
        )
        logger = pl.loggers.TensorBoardLogger(
            "lightning_logs", name="chessml", version=version_name
        )
        early_stop_callback = EarlyStopping(
            monitor="train_loss",
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode="min",
        )
        trainer = pl.Trainer(
            callbacks=[early_stop_callback], precision=16, logger=logger, max_epochs=100
        )
        model = EvaluationModel(
            batch_size=config["batch_size"],
            learning_rate=1e-3,
            layer_count=config["layer_count"],
        )

        trainer.fit(model)

        trainer.save_checkpoint(f"checkpoints/{version_name}.ckpt")
