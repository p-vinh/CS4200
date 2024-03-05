import numpy
import chess

# import tensorflow as tf
# import keras.models as load_model


# model = load_model.load_model(".\\checkpoints\\model.h5")

"""Return array of the best chess.Move

Keyword arguments:
current_board -- chess.Board()
model -- tf.saved_model
proportion -- proportion of best moves returned
"""
def find_best_moves(current_board, model, threshold=0.5):
    # Uses the model to predict the best moves
    # References the data frame to get the best moves
    pass


def get_piece_value(piece, square):
    x, y = square // 8, square % 8

    pass

def evaluate_board(board):
    """Return the evaluation of a board
    """
    evaluation = 0
    for square in chess.SQUARES:
        piece = str(board.piece_at(square))
        evaluation = evaluation + get_piece_value(piece, square)
    return evaluation


# Eval function from the model for the current position
# Replace with the model's eval function
def minimax_eval(board):
    with chess.engine.SimpleEngine.popen_uci(
        ".\\ChessML\\stockfish\\stockfish-windows-x86-64-avx2.exe"
    ) as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=3, time=0)).get("score")

        return result.black().score(mate_score=10000)


def minimax(board, depth, alpha, beta, maximizing_player):
    # if depth == 0:
    #     return -evaluate_board(board)
    # elif depth > 3:
    #     legal_moves = find_best_moves(board, model, 0.75)
    # else:
    #     legal_moves = list(board.legal_moves)

    if depth == 0 or board.is_game_over():
        return minimax_eval(board)

    if maximizing_player:
        max_eval = -numpy.inf
        for move in legal_moves:
            board.push(move)
            _eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, _eval)
            alpha = max(alpha, _eval)
            if beta <= alpha:
                return max_eval
        return max_eval
    else:
        min_eval = numpy.inf
        for move in legal_moves:
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
    # legal_moves = find_best_moves(board, model)
    max_eval = -numpy.inf
    max_move = None

    for move in board.legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, -numpy.inf, numpy.inf, False)
        board.pop()

        if value >= max_eval:
            max_eval = value
            max_move = move

    return max_move
