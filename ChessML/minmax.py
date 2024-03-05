import numpy as np
import chess
from collections import OrderedDict
from operator import itemgetter 
import tensorflow as tf
import pandas as pd
import model
import data_parser



model_chess = tf.keras.models.load_model("./ChessML/latest-model")

# Eval function from the model for the current position
# Replace with the model's eval function
def minimax_eval(board):
    board3d = data_parser.split_bitboard(board)
    board3d = np.expand_dims(board3d, 0)
    return model_chess.predict(board3d)[0][0]
    # with chess.engine.SimpleEngine.popen_uci(
    #     ".\\ChessML\\stockfish\\stockfish-windows-x86-64-avx2.exe"
    # ) as sf:
    #     result = sf.analyse(board, chess.engine.Limit(depth=3, time=0)).get("score")

    #     return result.black().score(mate_score=10000)


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
    legal_moves = model.find_best_moves(board, model)
    max_eval = -np.inf
    max_move = None

    for move in legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, -np.inf, np.inf, False)
        board.pop()

        if value >= max_eval:
            max_eval = value
            max_move = move

    return max_move
