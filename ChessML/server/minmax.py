import numpy as np
import chess
import data_parser
from model import EvaluationModel
import torch
import socket
from io import BytesIO

model_chess = EvaluationModel.load_from_checkpoint(
    "../checkpoints/M4batch_size-1024-layer_count-6.ckpt"
)

transposition_table = {}


# Eval function from the model for the current position
def minimax_eval(board):
    model_chess.eval()

    board = data_parser.split_bitboard(board)
    board = BytesIO(board)
    binary = np.frombuffer(board.getvalue(), dtype=np.uint8)
    board_tensor = torch.from_numpy(binary.copy()).to(torch.float32)

    with torch.no_grad():
        output = model_chess(board_tensor).item()
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
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", 8080))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            state = conn.recv(1024).decode()
            print('FEN Board', state)

            board = chess.Board(state)
            
            nb_moves = len(list(board.legal_moves))
            # best_move = minimax_root(board, 1)
            if nb_moves > 30:
                best_move = minimax_root(board, 4, False)
            elif nb_moves > 10 and nb_moves <= 30:
                best_move = minimax_root(board, 3, False)
            else:
                best_move = minimax_root(board, 5, False)
                
            print('Move: ', best_move)
            conn.sendall(best_move.uci().encode())
            
if __name__ == "__main__":
    while True:
        handle_game()
