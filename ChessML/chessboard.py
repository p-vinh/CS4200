import chess
import chess.pgn
import pygame as pg
import minmax
import time


WIDTH = HEIGHT = 400
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
font = pg.font.Font(None, 24)
clock = pg.time.Clock()
board = chess.Board()

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
light_square = (255, 206, 158)
dark_square = (209, 139, 71)


# Images
images = {
    "P": pg.image.load("./ChessML/pieces/wpawn.png"),
    "N": pg.image.load("./ChessML/pieces/wknight.png"),
    "B": pg.image.load("./ChessML/pieces/wbishop.png"),
    "R": pg.image.load("./ChessML/pieces/wrook.png"),
    "Q": pg.image.load("./ChessML/pieces/wqueen.png"),
    "K": pg.image.load("./ChessML/pieces/wking.png"),
    "p": pg.image.load("./ChessML/pieces/bpawn.png"),
    "n": pg.image.load("./ChessML/pieces/bknight.png"),
    "b": pg.image.load("./ChessML/pieces/bbishop.png"),
    "r": pg.image.load("./ChessML/pieces/brook.png"),
    "q": pg.image.load("./ChessML/pieces/bqueen.png"),
    "k": pg.image.load("./ChessML/pieces/bking.png"),
}


def drawBoard():
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                pg.draw.rect(
                    screen,
                    light_square,
                    pg.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE),
                )
            else:
                pg.draw.rect(
                    screen,
                    dark_square,
                    pg.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE),
                )


def drawPieces(board):
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece is not None:
                screen.blit(images[piece.symbol()], (col * SQ_SIZE, row * SQ_SIZE))


def ai_move(board):
    for move in board.legal_moves:
        future_board = chess.Board(board.fen())
        future_board.push(move)
        if future_board.is_checkmate():
            board.push(move)
            drawBoard()
            drawPieces(board)
            print(
                "Checkmate. {} wins".format(
                    "White" if board.turn == chess.BLACK else "Black"
                )
            )
            return
        elif future_board.is_stalemate():
            board.push(move)
            drawBoard()
            drawPieces(board)
            print("Stalemate. Neither player wins")
            return
        elif future_board.is_insufficient_material():
            board.push(move)
            drawBoard()
            drawPieces(board)
            print("Insufficient material. Neither player wins")
            return
    # board.push(minmax.minimax_root(board, 1))
    with chess.engine.SimpleEngine.popen_uci(
        ".\\ChessML\\stockfish\\stockfish-windows-x86-64-avx2.exe"
    ) as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=16, time=3  )).get("pv")
        board.push(result[0])
    # nb_moves = len(list(board.legal_moves))
    # if nb_moves > 30:
    #     board.push(minmax.minimax_root(board, 4))
    # elif nb_moves > 10 and nb_moves <= 30:
    #     board.push(minmax.minimax_root(board, 5))
    # else:
    #     board.push(minmax.minimax_root(board, 7))

    drawBoard()
    drawPieces(board)


def main():
    pg.display.set_caption("Chess")

    drawBoard()
    drawPieces(board)
    running = True
    sqSelected = ()
    playerClicks = []

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit()

            if board.turn == chess.WHITE:
                # if event.type == pg.MOUSEBUTTONDOWN:
                #     location = pg.mouse.get_pos()
                #     col = location[0] // SQ_SIZE
                #     row = location[1] // SQ_SIZE

                #     if sqSelected == (row, col):
                #         sqSelected = ()
                #         playerClicks = []
                #     else:
                #         sqSelected = (row, col)
                #         playerClicks.append(sqSelected)

                #     if len(playerClicks) == 2:
                #         move = chess.Move(
                #             chess.square(playerClicks[0][1], 7 - playerClicks[0][0]),
                #             chess.square(playerClicks[1][1], 7 - playerClicks[1][0]),
                #         )
                #         if board.piece_at(move.from_square) is not None:
                #             if (
                #                 board.piece_at(move.from_square).piece_type
                #                 == chess.PAWN
                #             ):
                #                 if move.to_square in chess.SquareSet(
                #                     chess.BB_RANK_1 | chess.BB_RANK_8
                #                 ):
                #                     move = chess.Move(
                #                         move.from_square,
                #                         move.to_square,
                #                         promotion=chess.QUEEN,
                #                     )
                #         if move in board.legal_moves:
                #             print(move)
                #             board.push(move)
                #         drawBoard()  # Redraw the board
                #         drawPieces(board)  # Update the pieces
                #         sqSelected = ()
                #         playerClicks = []
                ai_move(board)
            else:
                print("AI's turn")
                ai_move(board)

        clock.tick(60)
        pg.display.flip()

    time.sleep(5)


if __name__ == "__main__":
    main()
