"""
Write a program that can
place 8 queens in such a
manner on an 8 x 8
chessboard that no queens
attack each other by being in
the same row, column or
diagonal
"""

class EightQueens:
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.queens = 0

    def is_safe(self, row, col):
        pass

    def place_queen(self, col):
        pass

    def solve(self):
        pass

    def print_solution(self):
        pass
    
class Queen:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def potential_move(self):
        # return a list of potential moves
        
        # vertical
        for i in range(8):
            if i != self.row:
                yield (i, self.col)
        # horizontal
        for i in range(8):
            if i != self.col:
                yield (self.row, i)
        pass
    
    def __str__(self):
        return f"({self.row}, {self.col})"
    
    