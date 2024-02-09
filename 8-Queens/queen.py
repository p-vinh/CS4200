"""
Write a program that can
place 8 queens in such a
manner on an 8 x 8
chessboard that no queens
attack each other by being in
the same row, column or
diagonal
"""

class Eight_Queens:
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.queens = 0

    # Checks if a queen can be placed at the current position
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
        col = sum([self.board[i][col] for i in range(8)])
        if (col > 1):
            return False

        # check if there is a queen in the same row
        row = sum(self.board[row])
        if (row > 1):
            return False

        # check if there is a queen in the same diagonal
        for i in range(8):
            for j in range(8):
                if (i + j == row + col) or (i - j == row - col) and self.board[i][j] == 1:
                    return False
        return True

    # debug function
    def place_queen(self, row, col):
        self.board[row][col] = 1
        self.queens += 1


    def solve(self, col):
        self.board.__str__()

        if self.queens == 8:
            print("8 Queens have been placed")
            return True

        for row in range(8):
            if self.is_safe(row, col):
                self.board[row][col] = 1
                self.queens += 1

                if self.solve(col + 1):
                    return True

                self.board[row][col] = 0
                self.queens -= 1

        return False

    def __str__(self):
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.board)


if __name__ == "__main__":
    board = Eight_Queens()
    # Test cases for is_safe


    