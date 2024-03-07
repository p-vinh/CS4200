from peewee import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import time
from collections import OrderedDict
import data_parser
import numpy as np
from torch.utils.data import IterableDataset
import chess
from io import BytesIO

global model

class EvaluationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=10):
        super(EvaluationModel, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        layers = []
        
        layers.append((f"linear-{0}", nn.Linear(896, 896)))
        layers.append((f"relu-{0}", nn.ReLU()))
        layers.append((f"linear-{1}", nn.Linear(896, 448)))
        layers.append((f"relu-{1}", nn.ReLU()))
        layers.append((f"linear-{2}", nn.Linear(448, 224)))
        layers.append((f"relu-{2}", nn.ReLU()))
        layers.append((f"linear-{3}", nn.Linear(224, 112)))
        layers.append((f"relu-{3}", nn.ReLU()))
        layers.append((f"linear-{4}", nn.Linear(112, 1)))
        self.seq = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = x.view(-1, 896)
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.l1_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        dataset = data_parser.EvaluationDataset()
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True)
        


if __name__ == "__main__":
    configs = [
    {"layer_count": 4, "batch_size": 10},
    #  {"layer_count": 6, "batch_size": 1024},
    ]
    for config in configs:
        version_name = f'{int(time.time())}-batch_size-{config["batch_size"]}-layer_count-{config["layer_count"]}'
        logger = pl.loggers.TensorBoardLogger(
            "lightning_logs", name="chessml", version=version_name
        )
        trainer = pl.Trainer(precision=16, logger=logger, max_epochs=10000)
        model = EvaluationModel(
            batch_size=config["batch_size"],
            learning_rate=1e-3,
        )
        # trainer.tune(model)
        # lr_finder = trainer.tuner.lr_find(model, min_lr=1e-6, max_lr=1e-3, num_training=25)
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        trainer.fit(model)
        
        torch.save(model.state_dict(), f"./checkpoints/{version_name}.ckpt")
        
        break
    # 
    # board = chess.Board("8/8/8/8/7q/6r1/8/7K w - - 0 1")
    # print(minimax_eval(board))

