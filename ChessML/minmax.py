import numpy
import chess

# import tensorflow as tf
# import keras.models as load_model


# model = load_model.load_model(".\\checkpoints\\model.h5")


def get_all_moves(board):
    moves = list(board.legal_moves)
    for move in moves:
        if board.piece_at(move.from_square).piece_type == chess.PAWN:
            if move.to_square in chess.SquareSet(chess.BB_RANK_1 | chess.BB_RANK_8):
                yield chess.Move(
                    move.from_square, move.to_square, promotion=chess.QUEEN
                )
            else:
                yield move
        else:
            yield move


# Eval function from the model for the current position
def minimax_eval(board):
    with chess.engine.SimpleEngine.popen_uci(
        ".\\ChessML\\stockfish\\stockfish-windows-x86-64-avx2.exe"
    ) as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=3, time=0)).get("score")

        if result.is_mate():
            return numpy.inf if result.white().mate() > 0 else -numpy.inf
        else:
            return result.white().score(mate_score=10000)


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board)
    if maximizing_player:
        max_eval = -numpy.inf
        for move in get_all_moves(board):
            board.push(move)
            _eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, _eval)
            alpha = max(alpha, _eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = numpy.inf
        for move in get_all_moves(board):
            board.push(move)
            _eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, _eval)
            beta = min(beta, _eval)
            if beta <= alpha:
                break
        return min_eval


# Get the best move for the current position
def get_best_move(board, depth):
    max_move = None
    max_eval = -numpy.inf

    for move in get_all_moves(board):
        board.push(move)
        _eval = minimax(board, depth - 1, -numpy.inf, numpy.inf, False)
        board.pop()

        if _eval > max_eval:
            max_eval = _eval
            max_move = move
    return max_move
