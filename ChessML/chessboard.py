import chess
import chess.pgn
import numpy as np
import pygame as pg
import math
import random


board = chess.Board()

pg.init()
pg.display.set_caption("Chess")
width, height = 420, 420
offset = 20
screen = pg.display.set_mode((width, height))
clock = pg.time.Clock()

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
light_square = (255, 206, 158)
dark_square = (209, 139, 71)

# Fonts
font = pg.font.Font(None, 24)

# Images
images = {
    "P": pg.image.load("pieces/wpawn.png"),
    "N": pg.image.load("pieces/wknight.png"),
    "B": pg.image.load("pieces/wbishop.png"),
    "R": pg.image.load("pieces/wrook.png"),
    "Q": pg.image.load("pieces/wqueen.png"),
    "K": pg.image.load("pieces/wking.png"),
    "p": pg.image.load("pieces/bpawn.png"),
    "n": pg.image.load("pieces/bknight.png"),
    "b": pg.image.load("pieces/bbishop.png"),
    "r": pg.image.load("pieces/brook.png"),
    "q": pg.image.load("pieces/bqueen.png"),
    "k": pg.image.load("pieces/bking.png"),
}




def drawBoard():
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8"]

    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                pg.draw.rect(
                    screen,
                    light_square,
                    pg.Rect(col * 50 + offset, row * 50, 50, 50),
                )
            else:
                pg.draw.rect(
                    screen,
                    dark_square,
                    pg.Rect(col * 50 + offset, row * 50, 50, 50),
                )

                
            if col == 0:
                if board.turn:
                    text = font.render("", True, (255, 255, 255))
                    screen.blit(text, (col * 50 + 15, row * 50 + 30))
                    text = font.render(str(8 - row), True, (255, 255, 255))
                else:
                    # Clear text first
                    text = font.render("", True, (255, 255, 255))
                    screen.blit(text, (col * 50 + 15, row * 50 + 30))
                    text = font.render(numbers[row], True, (255, 255, 255))
                    
                screen.blit(text, (col * 50 + 15, row * 50 + 30))

letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
for row in range(8):
    for col in range(8):
        if row == 0:
            text = font.render(letters[col], True, (255, 255, 255))
            screen.blit(text, (col * 50 + 25, 400))

def drawPieces(board):
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece is not None:
                screen.blit(images[piece.symbol()], (col * 50 + offset, row * 50))

temp_move = ""



while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit()
  
        elif event.type == pg.KEYDOWN:
            print(board.legal_moves)
            if event.key == pg.K_RETURN:
                try:
                    board.push_san(temp_move)                    
                    temp_move = ""
                except:
                    temp_move = ""
            else:
                temp_move += event.unicode

    drawBoard()
    drawPieces(board)

    pg.display.flip()
    clock.tick(60)


