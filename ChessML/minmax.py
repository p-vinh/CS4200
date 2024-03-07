import numpy as np
import chess
from collections import OrderedDict
from operator import itemgetter 
# import tensorflow as tf
import data_parser
from model import EvaluationModel
import torch



model_chess = EvaluationModel.load_from_checkpoint(".\\checkpoints\\1709758613-batch_size-512-layer_count-4.ckpt")
# Eval function from the model for the current position
def minimax_eval(board):
    board = data_parser.split_bitboard(board)
    board_tensor = torch.from_numpy(board)
    
    board_tensor = board_tensor.unsqueeze(0)
    
    with torch.no_grad():
        return model_chess(board_tensor).item()
    


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
    board = chess.Board()
    print(minimax_eval(board))