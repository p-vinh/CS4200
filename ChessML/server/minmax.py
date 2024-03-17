import numpy as np
import chess
from model import EvaluationModel
import torch
import time
from multiprocessing import Pool
import socket

model_chess = EvaluationModel.load_from_checkpoint(
    ".\\checkpoints\\M3_batch_size-1024-layer_count-6.ckpt"
)

transposition_table = {}

squares_index = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]
    
def split_bitboard(board):
    # 1D array with 14 total, arrays of 64 bits
    # 14 arrays:
    # 6 arrays for white pieces
    # 6 arrays for black pieces
    # 2 arrays add attacks and valid moves so the network knows what is being attacked
    # concatenate the turn, castling rights, and en passant square to the end of the array
    # 903 total bits
    bitboards = np.array([], dtype=np.uint8)
    
    for piece in chess.PIECE_TYPES:
        bitboard = np.zeros(64, dtype=np.uint8)
        for square in board.pieces(piece, chess.WHITE):
            bitboard[square] = 1
        bitboards = np.append(bitboards, bitboard)
    for piece in chess.PIECE_TYPES:
        bitboard = np.zeros(64, dtype=np.uint8)
        for square in board.pieces(piece, chess.BLACK):
            bitboard[square] = 1
        bitboards = np.append(bitboards, bitboard)
    
    # Add attacks and valid moves
    aux = board.turn
    board.turn = chess.WHITE
    bitboard = np.zeros(64, dtype=np.uint8)
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        bitboard[i * 8 + j] = 1
    bitboards = np.append(bitboards, bitboard)
    
    board.turn = chess.BLACK
    bitboard = np.zeros(64, dtype=np.uint8)
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        bitboard[i * 8 + j] = 1
    bitboards = np.append(bitboards, bitboard)
    board.turn = aux

    bitboards = bitboards.reshape(14, 8, 8)
    binary = np.frombuffer(bitboards, dtype=np.uint8)
    return binary.tobytes()

# Eval function from the model for the current position
def minimax_eval(board):
    model_chess.eval()

    board = split_bitboard(board)
    binary = np.frombuffer(board, dtype=np.uint8).astype(np.float32)
    binary = binary.reshape(14, 8, 8)
    board_tensor = torch.from_numpy(binary)

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
            
            if nb_moves > 30:
                best_move = minimax_root_2(board, 4, 90)
            elif nb_moves > 10 and nb_moves <= 30:
                best_move = minimax_root_2(board, 5, 120)
            else:
                best_move = minimax_root_2(board, 7, 180)
                
            print('Move: ', best_move)
            conn.sendall(best_move.uci().encode())
            
if __name__ == "__main__":
    while True:
        handle_game()