import numpy as np
import chess
import data_parser
from model import EvaluationModel
import torch
import torch.nn.functional as F
from time import sleep


model_chess = EvaluationModel.load_from_checkpoint(".\\ChessML\\checkpoints\\V3-batch_size-20-layer_count-4.ckpt")

# Eval function from the model for the current position
def minimax_eval(board):
    with chess.engine.SimpleEngine.popen_uci(
        ".\\ChessML\\stockfish\\stockfish-windows-x86-64-avx2.exe"
        ) as sf:
            result = sf.analyse(board, chess.engine.Limit(depth=16)).get("score").white().score(mate_score=100) / 100

    board = data_parser.split_bitboard(board)
    binary = np.frombuffer(board, dtype=np.uint8).astype(np.float32)
    binary = binary.reshape(14, 8, 8)
    board_tensor = torch.from_numpy(binary)

    with torch.no_grad():
        threshold = 1.5
        output = model_chess(board_tensor).item() # Invert the output to match the stockfish output and play as black
        loss = abs(output - result)
        print(f"Model {output:2f}\nStockfish {result:2f}\nLoss {loss}")
        # if loss > threshold:
        #     return result
        # else:
        return output


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board)

    if maximizing_player:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            max_eval = max(max_eval, minimax(board, depth - 1, alpha, beta, not maximizing_player))
            board.pop()
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                return max_eval
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            min_eval = min(min_eval, minimax(board, depth - 1, alpha, beta, not maximizing_player))
            board.pop()
            beta = min(beta, min_eval)
            if beta <= alpha:
                return min_eval
        return min_eval

# Iterative deepening depth-first search: Combines depth-first search with breadth-first search
# Space complexity: O(bd)
# Time complexity: O(b^d)
# Depth-limited version of depth-first search, increasing the depth limit with each iteration until a solution is found
def iddfs(board, depth):
    for i in range(depth):
        result = dls(board, i)
        if result is not None:
            return result
    return None

def dls(board, depth):
    return minimax(board, depth, -np.inf, np.inf, False)

def minimax_root(board, depth):
    max_eval = -np.inf
    max_move = None

    for move in board.legal_moves:
        board.push(move)
        # value = minimax(board, depth - 1, -np.inf, np.inf, False)
        value = iddfs(board, depth)
        board.pop()

        if value is not None and value >= max_eval:
            max_eval = value
            max_move = move

    return max_move

if __name__ == "__main__":
    games = data_parser.test()
    
    for game in games:
        board = chess.Board(game[1])
        minimax_eval(board)
        print("FEN:", board.fen())
        print(board)