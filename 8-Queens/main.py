"""
Write a program that can
place 8 queens in such a
manner on an 8 x 8
chessboard that no queens
attack each other by being in
the same row, column or
diagonal
"""

class Board:
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.queens = 0

    def is_safe(self, row, col):
        # check if a queen can be placed at row, col
        # return True if it's safe, False otherwise

        # 3 Possible cases:
        # 1. Queen is in the midde of the board
        # 2. Queen is in the edge of the board
        # 3. Queen is in the corner of the board

        # check surrounding cells of the piece

        # check vertical

        # check horizontal

        # check diagonal

        
        return True
    def place_queen(self, col):

        # base case
        if col >= 8:
            return True

        pass

    def solve(self):
        pass

    def print_solution(self):
        pass

    def __str__(self):
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.board)

def main():
    board = Board()
    board.solve()
    board.print_solution()

if __name__ == "__main__":
    main()

    