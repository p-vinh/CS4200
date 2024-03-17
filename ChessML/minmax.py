import numpy as np
import chess
import data_parser
from model import EvaluationModel
import torch
import time
from multiprocessing import Pool
import socket

model_chess = EvaluationModel.load_from_checkpoint(
    ".\\checkpoints\\M3_batch_size-1024-layer_count-6.ckpt"
)

transposition_table = {}


# Eval function from the model for the current position
def minimax_eval(board):
    model_chess.eval()
    # result = data_parser.stock_fish_eval(board, 24)

    board = data_parser.split_bitboard(board)
    binary = np.frombuffer(board, dtype=np.uint8).astype(np.float32)
    binary = binary.reshape(14, 8, 8)
    board_tensor = torch.from_numpy(binary)

    with torch.no_grad():
        output = model_chess(board_tensor).item()
        # loss = abs(output - result)
        # print(f"Model {output:2f}\nStockfish {result:2f}\nLoss {loss}")
        return output

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board)

    board_hash = hash(board.fen()) # include the turn of the player in the hash

    if board_hash in transposition_table:
        return transposition_table[board_hash]

    if maximizing_player:
        max_eval = -9999
        for move in board.legal_moves:
            board.push(move)
            max_eval = max(
                max_eval, minimax(board, depth - 1, alpha, beta, not maximizing_player)
            )
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
            min_eval = min(
                min_eval, minimax(board, depth - 1, alpha, beta, not maximizing_player)
            )
            board.pop()
            beta = min(beta, min_eval)
            if beta <= alpha:
                return min_eval
        transposition_table[board_hash] = min_eval
        return min_eval

def evaluate_move(args):
    move, future_board, depth, maximizing_player = args
    result = minimax(
        chess.Board(future_board.fen()),
        depth - 1,
        -9999,
        9999,
        not maximizing_player,
    )
    return move, result

def minimax_root_2(board, depth, time_limit, maximizing_player=True):
    best_move = None
    best_value = -9999 if maximizing_player else 9999

    start_time = time.time()

    with Pool() as pool:
        args = []
        for move in board.legal_moves:
            future_board = chess.Board(board.fen())
            future_board.push(move)
            args.append((move, future_board, depth, maximizing_player))

        for move, result in pool.imap_unordered(evaluate_move, args):
            if time.time() - start_time > time_limit:
                print(f"Depth: {depth} Best move: {best_move} Value: {best_value}")
                break

            if result is not None:
                if maximizing_player and result >= best_value:
                    best_value = result
                    best_move = move
                elif not maximizing_player and result <= best_value:
                    best_value = result
                    best_move = move

    return best_move


def minimax_root(board, depth, maximizing_player=True):
    best_move = None
    best_value = -9999 if maximizing_player else 9999
    
    for move in board.legal_moves:
        future_board = chess.Board(board.fen())
        future_board.push(move)
        result = minimax(
            future_board, depth - 1, -9999, 9999, not maximizing_player
        )
        
        if result is not None:
            if maximizing_player and result >= best_value:
                best_value = result
                best_move = move
            elif not maximizing_player and result <= best_value:
                best_value = result
                best_move = move
            
    return best_move

def handle_game():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 1234))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            state = conn.recv(1024).decode()
            best_move = minimax_root(state, 4, True)
            
            conn.sendall(best_move.encode())
            
if __name__ == "__main__":
    while True:
        handle_game()