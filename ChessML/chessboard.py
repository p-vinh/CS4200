import chess
import chess.pgn
import pygame as pg
import minmax


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
            elif event.type == pg.MOUSEBUTTONDOWN:
                if board.turn == chess.WHITE:
                    location = pg.mouse.get_pos()
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    
                    if sqSelected == (row, col):
                        sqSelected = ()
                        playerClicks = []
                    else:
                        sqSelected = (row, col)
                        playerClicks.append(sqSelected)
                    
                    if len(playerClicks) == 2:
                        move = chess.Move(
                            chess.square(playerClicks[0][1], 7 - playerClicks[0][0]),
                            chess.square(playerClicks[1][1], 7 - playerClicks[1][0]),
                        )
                        if move in board.legal_moves:
                            print(move)
                            board.push(move)
                            drawBoard() # Redraw the board
                            drawPieces(board) # Update the pieces
                        sqSelected = ()
                        playerClicks = []
                else:
                    move = minmax.get_best_move(board, 4)
                    board.push(move)
                    drawBoard()
                    drawPieces(board)
                
        clock.tick(60)
        pg.display.flip()



if __name__ == "__main__":
    main()


