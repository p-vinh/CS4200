import numpy as np
import chess
import data_parser
from model import EvaluationModel
import torch
import torch.nn.functional as F
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import random

model_chess = EvaluationModel.load_from_checkpoint(".\\ChessML\\checkpoints\\epoch=210-step=26375.ckpt")

# Eval function from the model for the current position
def minimax_eval(board):
    # result = stock_fish_eval(board, 16)

    board = data_parser.split_bitboard(board)
    binary = np.frombuffer(board, dtype=np.uint8).astype(np.float32)
    binary = binary.reshape(14, 8, 8)
    board_tensor = torch.from_numpy(binary)

    with torch.no_grad():
        # threshold = 1.5
        output = model_chess(board_tensor).item()
        # loss = abs(output - result)
        # print(f"Model {output:2f}\nStockfish {result:2f}\nLoss {loss}")
        # if loss > threshold:
        #     return result
        # else:
        return output

def stock_fish_eval(board, depth):
    with chess.engine.SimpleEngine.popen_uci(
        ".\\ChessML\\stockfish\\stockfish-windows-x86-64-avx2.exe"
    ) as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth)).get("score")
        # print(board)
        
        if result.white().is_mate():
            if result.white().mate() > 0:
                return 15
            else:
                return -15
            
        eval = result.white().score() / 100
        return eval
    
transposition_table = {}

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board)

    board_hash = hash(board.fen())

    if board_hash in transposition_table:
        return transposition_table[board_hash]

    if maximizing_player:
        max_eval = -9999
        for move in board.legal_moves:
            board.push(move)
            max_eval = max(max_eval, minimax(board, depth - 1, alpha, beta, not maximizing_player))
            board.pop()
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                return max_eval
        transposition_table[board_hash] = max_eval
        return max_eval
    else:
        min_eval = 9999
        for move in board.legal_moves:
            board.push(move)
            min_eval = min(min_eval, minimax(board, depth - 1, alpha, beta, not maximizing_player))
            board.pop()
            beta = min(beta, min_eval)
            if beta <= alpha:
                return min_eval
        transposition_table[board_hash] = min_eval
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
    return minimax(board, depth, -9999, 9999, False)

def minimax_root(board, depth, maximizing_player=True):
    with ThreadPoolExecutor() as executor:
        futures = []
        moves = sorted(board.legal_moves, key=lambda move: board.is_capture(move), reverse=True)
        for move in moves:
            future_board = chess.Board(board.fen())
            future_board.push(move)
            futures.append((move, executor.submit(minimax, future_board, depth - 1, -9999, 9999, not maximizing_player)))
        results = [(move, future.result()) for move, future in futures]

    best_move = max(results, key=lambda x: x[1])[0]
    return best_move
    # max_eval = -np.inf
    # max_move = None

    # for move in board.legal_moves:
    #     board.push(move)
    #     value = minimax(board, depth - 1, -9999, 9999, False)
    #     # value = iddfs(board, depth)
    #     board.pop()

    #     if value is not None and value >= max_eval:
    #         max_eval = value
    #         max_move = move

    # return max_move

if __name__ == "__main__":
    # games = data_parser.test()
    # for game in games:
    #     board = chess.Board(game[1])
    #     print(minimax_eval(board))

    board = chess.Board()
    while board.is_game_over() == False:
        board.push(minimax_root(board, 4, True))
        print("FEN:", board.fen())
        print(board)
        move = minimax_root(board, 4, False)

        board.push(move)

        print("Post-move")
        print(board)