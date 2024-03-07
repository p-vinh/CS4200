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

global model
class EvaluationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=1024, layer_count=10):
        super(EvaluationModel, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        layers = []
        for i in range(layer_count - 1):
            layers.append((f"linear-{i}", nn.Linear(3640, 3640)))
            layers.append((f"relu-{i}", nn.ReLU()))
        layers.append((f"linear-{layer_count-1}", nn.Linear(3640, 1)))
        self.seq = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
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
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn
        )
        
def collate_fn(data):
    bin = [torch.from_numpy(item['binary']) for item in data]
    eval = [torch.from_numpy(item['eval']) for item in data]

    max_len = max([tensor.shape[0] for tensor in bin])
    bin = [F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in bin]
    
    binary = torch.stack(bin)
    eval = torch.stack(eval)
    print(binary.shape, eval.shape)
    return binary, eval



# Eval function from the model for the current position
def minimax_eval(board):
    board = data_parser.split_bitboard(board)
    board_tensor = torch.from_numpy(board)
    
    board_tensor = board_tensor.unsqueeze(0)
    
    with torch.no_grad():
        return model(board_tensor).item()
    


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board)


    if depth == 0 or board.is_game_over():
        return minimax_eval(board)

    if maximizing_player:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            _eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, _eval)
            alpha = max(alpha, _eval)
            if beta <= alpha:
                return max_eval
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            _eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, _eval)
            beta = min(beta, _eval)
            if beta <= alpha:
                return min_eval
        return min_eval


def minimax_root(board, depth):
    # Searching for the top 50% best moves. Restricts the search space
    max_eval = -np.inf
    max_move = None

    for move in board.legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, -np.inf, np.inf, False)
        board.pop()

        if value >= max_eval:
            max_eval = value
            max_move = move

    return max_move

if __name__ == "__main__":
    configs = [
    {"layer_count": 4, "batch_size": 512},
    #  {"layer_count": 6, "batch_size": 1024},
    ]
    for config in configs:
        version_name = f'{int(time.time())}-batch_size-{config["batch_size"]}-layer_count-{config["layer_count"]}'
        logger = pl.loggers.TensorBoardLogger(
            "lightning_logs", name="chessml", version=version_name
        )
        trainer = pl.Trainer(precision=16, logger=logger, max_epochs=10)
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
        
        trainer.save_checkpoint(f"checkpoints/{version_name}.ckpt")
        
    #     break
    model = EvaluationModel.load_from_checkpoint(".\\checkpoints\\1709758613-batch_size-512-layer_count-4.ckpt")
    board = chess.Board()
    print(minimax_eval(board))

