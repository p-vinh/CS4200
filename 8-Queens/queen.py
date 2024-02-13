"""
Write a program that can
place 8 queens in such a
manner on an 8 x 8
chessboard that no queens
attack each other by being in
the same row, column or
diagonal
"""
import pygame
import time
from pygame.locals import *

SQUARE_SIZE = 50
GRID_SIZE = 8
QUEEN = pygame.transform.scale(pygame.image.load('.\\images\\chess-queen.svg'), (SQUARE_SIZE, SQUARE_SIZE))

LIGHT_BROWN = (164,124,72)
BROWN = (139, 69, 19)
WHITE = (255, 255, 255)

pygame.init()
pygame.font.init()
pygame.display.set_caption('8-Queens')

screen_width = GRID_SIZE * SQUARE_SIZE + 200
screen_height = GRID_SIZE * SQUARE_SIZE
screen = pygame.display.set_mode((screen_width, screen_height))
font = pygame.font.Font(None, 36)

class Eight_Queens:
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.queens = 0
        self.skip = False

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

    # 2 Different ways to solve the 8 queens problem
    def solve_col(self, col):
        # GUI Logic
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if button.collidepoint(event.pos):
                        self.skip = True
                        
        # base case
        if self.queens == 8:
            self.draw_board()
            print(self)
            return True

        for row in range(8):
            if self.is_safe(row, col):
                self.board[row][col] = 1
                self.queens += 1
        
                if (not self.skip):
                    self.draw_board()
                    time.sleep(0.5)
                
                if self.solve_col(col + 1):
                    return True

                # backtrack if solve returns False
                self.board[row][col] = 0
                self.queens -= 1

            self.draw_board()

        return False
    
    def solve_row(self, row):
        # GUI Logic
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if button.collidepoint(event.pos):
                        self.skip = True
                        
        # base case
        if self.queens == 8:
            self.draw_board()
            print(self)
            return True

        for col in range(8):
            if self.is_safe(row, col):
                self.board[row][col] = 1
                self.queens += 1
        
                if (not self.skip):
                    self.draw_board()
                    time.sleep(0.2)
                
                if self.solve_row(row + 1):
                    return True

                # backtrack if solve returns False
                self.board[row][col] = 0
                self.queens -= 1

            self.draw_board()

        return False


    def draw_board(self):
        self.reset()
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.board[x][y] == 1:
                    screen.blit(QUEEN, (y*SQUARE_SIZE + 6, x*SQUARE_SIZE + 5))
        pygame.display.flip()
        
    def reset(self):
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x*SQUARE_SIZE, y*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(screen, LIGHT_BROWN if (x+y) % 2 == 0 else BROWN, rect)
                


    def __str__(self):
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.board)




if __name__ == "__main__":
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x*SQUARE_SIZE, y*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, LIGHT_BROWN if (x+y) % 2 == 0 else BROWN, rect)
    
    button_x = screen_width - 130
    button_y = screen_height - 150

    button = pygame.Rect(button_x, button_y, 100, 50)
    text = font.render('Skip', True, WHITE)
    screen.blit(text, (button_x, button_y))
        
    pygame.display.flip()

    solver = Eight_Queens()
    print(solver.solve_row(0))
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        pygame.display.flip()
                        
