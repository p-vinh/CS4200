import chess
import numpy as np
import pygame as pg


board = chess.Board()

pg.init()
pg.display.set_caption("Chess")
width, height = 400, 400
screen = pg.display.set_mode((width, height))
clock = pg.time.Clock()

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
light_square = (255, 206, 158)
dark_square = (209, 139, 71)

# Fonts
font = pg.font.Font(None, 36)

# Images
pieces = {
    "P": pg.image.load("pieces/wp.png"),
    "N": pg.image.load("pieces/wknight.png"),
    "B": pg.image.load("pieces/wbishop.png"),
    "R": pg.image.load("pieces/wrook.png"),
    "Q": pg.image.load("pieces/wqueen.png"),
    "K": pg.image.load("pieces/wking.png"),
    "p": pg.image.load("pieces/bp.png"),
    "n": pg.image.load("pieces/bknight.png"),
    "b": pg.image.load("pieces/bbishop.png"),
    "r": pg.image.load("pieces/brook.png"),
    "q": pg.image.load("pieces/bqueen.png"),
    "k": pg.image.load("pieces/bking.png"),
}

image_size = pieces["P"].get_size()

def drawBoard():
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                pg.draw.rect(
                    screen,
                    light_square,
                    pg.Rect(col * 50, row * 50, 50, 50),
                )
            else:
                pg.draw.rect(
                    screen,
                    dark_square,
                    pg.Rect(col * 50, row * 50, 50, 50),
                )
def drawPieces(board):
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece is not None:
                if piece.symbol() == "P" or piece.symbol() == "p":
                    screen.blit(pg.transform.scale(pieces[piece.symbol()], (image_size[0] + 2, image_size[1] + 2)), (col * 50, row * 50))
                else:
                    screen.blit(pieces[piece.symbol()], (col * 50, row * 50))

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit()
    drawBoard()
    drawPieces(board)
    pg.display.flip()
    clock.tick(60)   
print(board)


def checkEndCondition(board):
    if (
        board.is_checkmate()
        or board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_threefold_repetition()
        or board.can_claim_fifty_moves()
        or board.can_claim_draw()
    ):
        return True
    return False
