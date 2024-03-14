from peewee import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from collections import OrderedDict
import data_parser
import numpy as np
import chess

global model
 
class EvaluationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=1024):
        super(EvaluationModel, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        layers = []
        
        layers.append((f"linear-{0}", nn.Linear(896, 896))) # 14 * 8 * 8
        layers.append((f"relu-{0}", nn.ReLU()))
        layers.append((f"linear-{1}", nn.Linear(896, 448))) # 14 * 8 * 8 -> 14 * 4 * 8
        layers.append((f"relu-{1}", nn.ReLU()))
        layers.append((f"linear-{2}", nn.Linear(448, 224))) # 14 * 4 * 8 -> 14 * 4 * 4
        layers.append((f"relu-{2}", nn.ReLU()))
        layers.append((f"linear-{3}", nn.Linear(224, 112))) # 14 * 4 * 4 -> 7 * 4 * 4
        layers.append((f"relu-{3}", nn.ReLU()))
        layers.append((f"linear-{4}", nn.Linear(112, 1))) # 7 * 4 * 4 -> 1
        self.seq = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = x.view(-1, 896)
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.l1_loss(y_hat, y)
        print("loss", loss)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        dataset = data_parser.EvaluationDataset()
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)
        


if __name__ == "__main__":
    configs = [
    {"layer_count": 4, "batch_size": 1024},
    #  {"layer_count": 6, "batch_size": 1024},
    ]
    for config in configs:
        version_name = f'V2batch_size-{config["batch_size"]}-layer_count-{config["layer_count"]}'
        logger = pl.loggers.TensorBoardLogger(
            "lightning_logs", name="chessml", version=version_name
        )
        trainer = pl.Trainer(precision=16, logger=logger, max_epochs=300)
        model = EvaluationModel(
            batch_size=config["batch_size"],
            learning_rate=1e-3,
        )
        # trainer.tune(model)
        # lr_finder = trainer.tuner.lr_find(model, min_lr=1e-6, max_lr=1e-3, num_training=25)
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        trainer.fit(model)
        
        trainer.save_checkpoint(f"checkpoints/{version_name}.ckpt")
        
        break
    # model = EvaluationModel.load_from_checkpoint(".\\checkpoints\\1709758613-batch_size-512-layer_count-4.ckpt")
    # board = chess.Board()
    # print(minimax_eval(board))

