import numpy as np
import chess
import data_parser
from model import EvaluationModel
import torch
import torch.nn.functional as F
from time import sleep


model_chess = EvaluationModel.load_from_checkpoint(".\\ChessML\\checkpoints\\epoch=210-step=26375.ckpt")

# Eval function from the model for the current position
def minimax_eval(board):
    with chess.engine.SimpleEngine.popen_uci(
        ".\\ChessML\\stockfish\\stockfish-windows-x86-64-avx2.exe"
        ) as sf:
            result = sf.analyse(board, chess.engine.Limit(depth=16)).get("score").black().score(mate_score=10000) / 100

    board = data_parser.split_bitboard(board)
    binary = np.frombuffer(board, dtype=np.uint8).astype(np.float32)
    binary = binary.reshape(14, 8, 8)
    board_tensor = torch.from_numpy(binary)

    with torch.no_grad():
        threshold = 1
        output = model_chess(board_tensor).item()
        loss = F.l1_loss(torch.tensor([output]), torch.tensor([result])).item()

        if loss > threshold:
            return result
        else:
            # print(f'Eval {result} Prediction {output:.2f} Loss {loss:.2f}')
            return output


def minimax(board, depth, alpha, beta, maximizing_player):
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
    for i in range(10):
        game = data_parser.test()
        board = chess.Board(game[1])
        minimax_eval(board)
        print("FEN:", board.fen())
        print(board)
        sleep(1)