import numpy
import chess
import tensorflow as tf
import keras.models as load_model

square_to_index = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
}
model = load_model.load_model(".\\checkpoints\\model.h5")


def square_to_coordinates(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), square_to_index[letter[0]]


def split_dims(board):
    board3d = numpy.zeros((14, 8, 8), dtype=numpy.int8)

    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    aux = board.turn
    board.turn = chess.WHITE

    for move in board.legal_moves:
        i, j = square_to_coordinates(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_coordinates(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux

    return board3d


# Eval function from the model for the current position
def minimax_eval(board):
    board3d = split_dims(board)
    board3d = numpy.expand_dims(board3d, axis=0)
    return model.predict(board3d)[0][0]


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board)
    if maximizing_player:
        max_eval = -numpy.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = numpy.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


# Get the best move for the current position
def get_best_move(board, depth):
    max_move = None
    max_eval = -numpy.inf

    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -numpy.inf, numpy.inf, False)
        board.pop()

        if eval > max_eval:
            max_eval = eval
            max_move = move
    return max_move
