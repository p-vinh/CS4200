import numpy as np
import chess
import data_parser
from model import EvaluationModel
import torch
import time
from multiprocessing import Pool
import boto3
import json
import os

model_chess = EvaluationModel.load_from_checkpoint(
    ".\\checkpoints\\V2batch_size-1024-layer_count-4.ckpt"
)


# Eval function from the model for the current position
def minimax_eval(board):
    model_chess.eval()
    # result = data_parser.stock_fish_eval(board, 16)

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

# Iterative deepening depth-first search: Combines depth-first search with breadth-first search
# Space complexity: O(bd)
# Time complexity: O(b^d)
# Depth-limited version of depth-first search, increasing the depth limit with each iteration until a solution is found
# or a time limit is reached
def minimax_root(board, depth, time_limit, maximizing_player=True):
    best_move = None
    best_value = -9999 if maximizing_player else 9999

    start_time = time.time()

    with Pool() as pool:
        for depth in range(1, depth + 1):
            if time.time() - start_time > time_limit:
                print(f"Depth: {depth} Best move: {best_move} Value: {best_value}")
                break
            
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
                    if maximizing_player and result > best_value:
                        best_value = result
                        best_move = move
                    elif not maximizing_player and result < best_value:
                        best_value = result
                        best_move = move

    return best_move


if __name__ == "__main__":
    # games = data_parser.test()
    # for game in games:
    #     board = chess.Board(game[1])
    #     print(minimax_eval(board))

    # board = chess.Board()
    # while board.is_game_over() == False:
    #     move = minimax_root(board, 4, 120, True)
    #     board.push(move)
    #     print("FEN:", board.fen())
    #     print(board)
    #     move = minimax_root(board, 4, 120, False)
    #     board.push(move)
    #     print("Post-move")
    #     print("FEN:", board.fen())
    #     print(board)
    # print(f"Player {board.result()} won the game!")
    
    board = chess.Board()
    client = boto3.client('lambda', region_name='us-west-2')
    
    response = client.invoke(
        FunctionName='minimax',
        InvocationType='RequestResponse',
        Payload=json.dumps({
            "board": board.fen(),
            "depth": 4,
            "time_limit": 120,
            "maximizing_player": True  
        })
    )
    
    result = json.loads(response['Payload'].read())
    print(result)
