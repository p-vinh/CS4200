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
        # solve the 8-queens problem using breadth-first search
        fringe = []
        fringe.append((0,0))    

        while fringe:
            # pop the first element in the fringe (queue)
            row, col = fringe.pop(0)
            
            # place a queen at the current position if it's safe
            if self.is_safe(row, col):

                # 1 represents a queen and increment the number of queens
                self.board[row][col] = 1
                self.queens += 1

                # place the next queen in the next column if it's safe then append the position to the fringe
                if self.place_queen(col+1):
                    results.append((row, col))
                self.board.__str__()
                # remove the queen and decrement the number of queens if the next queen can't be placed
                self.board[row][col] = 0
                self.queens -= 1
            if self.queens < 8:
                if row < 7:
                    fringe.append((row+1, col))
                if col < 7:
                    fringe.append((row, col+1))
            else: # 8 queens have been placed
                break


        return fringe

    def __str__(self):
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.board)


if __name__ == "__main__":
    board = Eight_Queens()
    board.solve()

    