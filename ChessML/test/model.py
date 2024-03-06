from peewee import *
import base64

db = SqliteDatabase("2021-07-31-lichess-evaluations-37MM.db")


class Evaluations(Model):
    id = IntegerField()
    fen = TextField()
    binary = BlobField()
    eval = FloatField()

    class Meta:
        database = db

    def binary_base64(self):
        return base64.b64encode(self.binary)


db.connect()
LABEL_COUNT = 37164639
print(LABEL_COUNT)
eval = Evaluations.get(Evaluations.id == 1)
print(eval.binary_base64())

import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
import pytorch_lightning as pl
from random import randrange


class EvaluationDataset(IterableDataset):
    def __init__(self, count):
        self.count = count

    def __iter__(self):
        return self

    def __next__(self):
        idx = randrange(self.count)
        return self[idx]

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        eval = Evaluations.get(Evaluations.id == idx + 1)
        bin = np.frombuffer(eval.binary, dtype=np.uint8)
        bin = np.unpackbits(bin, axis=0).astype(np.single)
        eval.eval = max(eval.eval, -15)
        eval.eval = min(eval.eval, 15)
        ev = np.array([eval.eval]).astype(np.single)
        return {"binary": bin, "eval": ev}


dataset = EvaluationDataset(count=LABEL_COUNT)


import time
from collections import OrderedDict


class EvaluationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=1024, layer_count=10):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        layers = []
        for i in range(layer_count - 1):
            layers.append((f"linear-{i}", nn.Linear(808, 808)))
            layers.append((f"relu-{i}", nn.ReLU()))
        layers.append((f"linear-{layer_count-1}", nn.Linear(808, 1)))
        self.seq = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["binary"], batch["eval"]
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        dataset = EvaluationDataset(count=LABEL_COUNT)
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True
        )


configs = [
    {"layer_count": 4, "batch_size": 512},
    #  {"layer_count": 6, "batch_size": 1024},
]
for config in configs:
    version_name = f'{int(time.time())}-batch_size-{config["batch_size"]}-layer_count-{config["layer_count"]}'
    logger = pl.loggers.TensorBoardLogger(
        "lightning_logs", name="chessml", version=version_name
    )
    trainer = pl.Trainer(precision=16, max_epochs=1, logger=logger)
    model = EvaluationModel(
        layer_count=config["layer_count"],
        batch_size=config["batch_size"],
        learning_rate=1e-3,
    )
    # trainer.tune(model)
    # lr_finder = trainer.tuner.lr_find(model, min_lr=1e-6, max_lr=1e-3, num_training=25)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    trainer.fit(model)
    break


def svg_url(fen):
    fen_board = fen.split()[0]
    return SVG_BASE_URL + fen_board


def show_index(idx):
    eval = Evaluations.select().where(Evaluations.id == idx + 1).get()
    batch = dataset[idx]
    x, y = torch.tensor(batch["binary"]), torch.tensor(batch["eval"])
    y_hat = model(x)
    loss = F.l1_loss(y_hat, y)
    print(
        f"Idx {idx} Eval {y.data[0]:.2f} Prediction {y_hat.data[0]:.2f} Loss {loss:.2f}"
    )
    print(f"FEN {eval.fen}")
    display(SVG(url=svg_url(eval.fen)))


for i in range(5):
    idx = randrange(LABEL_COUNT)
    show_index(idx)
