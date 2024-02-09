"""
Write a program that can
place 8 queens in such a
manner on an 8 x 8
chessboard that no queens
attack each other by being in
the same row, column or
diagonal
"""
import unittest
import pygame

class Eight_Queens:
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.queens = 0

    def is_safe(self, row, col):
        # 3 Possible cases:
        # 1. Queen is in the midde of the board
        # 2. Queen is in the edge of the board
        # 3. Queen is in the corner of the board
    
        # check surrounding cells of the current cell for a queen
        surrounding_cells = [(row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col), (row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)]

        for x, y in surrounding_cells:
            if 0 <= x < 8 and 0 <= y < 8:
                if self.board[x][y] == 1:
                    return False
        
        # check if there is a queen in the same column
        for i in range(8):
            if (self.board[i][col] == 1):
                return False

        # check if there is a queen in the same row
        for i in range(8):
            if (self.board[row][i] == 1):
                return False

        # upper diagonal left
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if self.board[i][j] == 1:
                return False
        # upper diagonal right
        for i, j in zip(range(row, -1, -1), range(col, 8, 1)):
            if self.board[i][j] == 1:
                return False

        # lower diagonal right
        for i, j in zip(range(row, 8, 1), range(col, 8, 1)):
            if self.board[i][j] == 1:
                return False
        
        # lower diagonal left
        for i, j in zip(range(row, 8, 1), range(col, -1, -1)):
            if self.board[i][j] == 1:
                return False

        return True

    # debug function
    def place_queen(self, row, col):
        self.board[row][col] = 1
        self.queens += 1


    def solve(self, col):
        if self.queens == 8:
            print("8 Queens have been placed")
            return self.get_all_positions()

        for row in range(8):
            if self.is_safe(row, col):
                self.board[row][col] = 1
                self.queens += 1

                if self.solve(col + 1):
                    return True

                self.board[row][col] = 0
                self.queens -= 1

        return False

    def get_all_positions(self):
        return [(i, j) for i in range(8) for j in range(8) if self.board[i][j] == 1]

    def __str__(self):
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.board)


SQUARE_SIZE = 50
GRID_SIZE = 8
QUEEN = pygame.transform.scale(pygame.image.load('8-Queens\images\chess-queen.svg'), (SQUARE_SIZE, SQUARE_SIZE))
queen_width, queen_height = QUEEN.get_size()

WHITE = (255, 255, 255)
BROWN = (139, 69, 19)

pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * SQUARE_SIZE, GRID_SIZE * SQUARE_SIZE))

for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        rect = pygame.Rect(x*SQUARE_SIZE, y*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.rect(screen, WHITE if (x+y) % 2 == 0 else BROWN, rect)

pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        solver = Eight_Queens()
        solver.solve(0)
    if pygame.key.get_pressed()[pygame.K_SPACE]:
         
        for x, y in solver.get_all_positions():
            screen.blit(QUEEN, (x*SQUARE_SIZE + 6, y*SQUARE_SIZE + 5))

        pygame.display.flip()

pygame.quit()

