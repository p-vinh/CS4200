import chess.pgn
import chess.engine
import numpy
import chess
import pymysql
import sys
import os
import traceback
import binascii
from dotenv import load_dotenv

"""
EvaluationDataset takes single random row from the SQLite table and preprocesses it by extracting the
binary value in raw bytes, converting those bytes to floats using numpy’s frombuffer and unpackbits functions,
and forming the required 808 length float array as input. The evaluation value is is extracted and bounded between -15 and 15.
"""
DEPTH = 16


# ======================AWS RDS MySQL Connection===============================
class EvaluationDataset:

    def __init__(self):
        load_dotenv()
        self.endpoint = os.environ.get("DB_ENDPOINT")
        self.port = os.environ.get("DB_PORT")
        self.user = os.environ.get("DB_USER")
        self.password = os.environ.get("DB_PASS")
        self.region = os.environ.get("DB_REGION")
        self.dbname = os.environ.get("DB_NAME")

        print("Connecting to MySQL database")
        self.db = self.connect()
        self.cursor = self.db.cursor()
        self.cursor.execute("CREATE DATABASE IF NOT EXISTS chessai")
        self.cursor.connection.commit()
        self.cursor.execute("USE chessai")
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS ChessData (id INT NOT NULL, fen VARCHAR(100) NOT NULL, bin BLOB NOT NULL, eval FLOAT NOT NULL, PRIMARY KEY (id))"
        )
        self.cursor.execute("SHOW TABLES")
        print("Current database: ", self.cursor.fetchall())

    def __next__(self):
        try:
            self.cursor = self.db.cursor()
            self.cursor.execute("SELECT * FROM ChessData ORDER BY RAND() LIMIT 1")
            eval = self.cursor.fetchone()
            bin = numpy.frombuffer(eval.binary, dtype=numpy.uint8)
            bin = numpy.unpackbits(bin).astype(numpy.single)
            print(bin)
            eval.eval = max(eval.eval, -15)
            eval.eval = min(eval.eval, 15)
            ev = numpy.array([eval.eval]).astype(numpy.single)
            print(ev)
            return {"binary": bin, "eval": ev}
        except Exception as e:
            print("Database connection failed due to {}".format(e))

    def connect(self):
        try:
            conn = pymysql.connect(
                host=self.endpoint, user=self.user, password=self.password
            )
            if conn:
                print("Database connection successful")
            return conn
        except Exception as e:
            print("Database connection failed due to {}".format(e))

    def import_game(self, pgn_file):
        try:
            self.cursor = self.db.cursor()
            game_id = 1
            with open(pgn_file) as pgn:
                game = chess.pgn.read_game(pgn)                
                while game is not None:
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        eval = self.stock_fish_eval(board, DEPTH)
                        binary = split_bitboard(board)

                        print("Inserting into database: ", game_id, eval)
                        if eval is not None:
                            self.cursor.execute(
                                "INSERT INTO ChessData (id, fen, bin, eval) VALUES (%s, %s, %s, %s)",
                                (game_id, board.fen(), binary, eval),
                            )
                        else:
                            print("No evaluation found for game: ", game_id)
                            break
                        game_id += 1
                    self.db.commit()
                    game = chess.pgn.read_game(pgn)
            self.cursor.close()
        except Exception as e:
            print("Error inserting into database: {}".format(e))
            traceback.print_exc()

    def __getitem__(self, idx):
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute("SELECT * FROM your_table WHERE id = %s", (idx,))
            eval = cur.fetchone()
            bin = numpy.frombuffer(eval.binary, dtype=numpy.uint8)
            bin = numpy.unpackbits(bin, axis=0).astype(numpy.single)

            eval.eval = max(eval.eval, -15)
            eval.eval = min(eval.eval, 15)
            ev = numpy.array([eval.eval]).astype(numpy.single)

            cur.close()
            conn.close()

            return {"binary": bin, "eval": ev}

        except Exception as e:
            print("Database connection failed due to {}".format(e))

    # Evaluate the board using Stockfish: Positive score means white is winning, negative score means black is winning
    def stock_fish_eval(self, board, depth):
        with chess.engine.SimpleEngine.popen_uci(
            ".\\ChessML\\stockfish\\stockfish-windows-x86-64-avx2.exe"
        ) as sf:
            result = sf.analyse(board, chess.engine.Limit(depth=depth)).get("score")
            print(board)
            return result.white().score(mate_score=10000)

    def delete(self):
        try:
            self.cursor = self.db.cursor()
            self.cursor.execute("DELETE FROM ChessData")
            self.db.commit()
            self.cursor.close()
            self.db.close()
        except Exception as e:
            print("Database connection failed due to {}".format(e))

    def close(self):
        try:
            self.db.close()
        except pymysql.err.Error as e:
            if str(e) != "Already closed":
                raise

    # ==========================================================================


squares_index = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]
    
def split_bitboard(board):
    # 3D array with 14 layers, 8 rows, and 8 columns
    # 14 Layers:
    # 6 Layers for white pieces
    # 6 Layers for black pieces
    # 2 Layers add attacks and valid moves so the network knows what is being attacked
    board3d = numpy.zeros((14, 8, 8), dtype=numpy.int8)
    
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1
    
    # Add attacks and valid moves
    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux
    
    return board3d


def test():
    try:


        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM ChessData")
        count = cur.fetchone()[0]
        print(f"Number of rows in ChessData: {count}")

        cur.execute("SELECT * FROM ChessData WHERE fen=\"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\"")
        row = cur.fetchone()
        # for row in rows:
        s = row[3].decode("utf-8")
        s = s.replace(" ", "")
        s = s.replace("\n", "")
        s = s.replace("\t", "")
        s = s.replace("[", "")
        s = s.replace("]", "")

        for i in range(0, len(s), 8):
            print(s[i:i+8])
        # # TODO: Convert the binary string to a 3D numpy array
        binary_string = bin(int.from_bytes(s, byteorder="big"))[2::]
        print(binary_string)
        # numpy.array

        # print(binary_string)
        # lst = [list(map(int, sublist.split())) for sublist in s.split('\n')]

        # arr = n/umpy.array(lst)
        # arr = arr.reshape(14, 8, 8)
        # bins.append(arr)


        # for row in rows:
        #     print("Game ID: ", row[0])
        #     print("FEN: ", row[1])
        #     print("Evaluation: ", row[2])
        #     print("Binary: ", row[3])
        #     print("")

        # boad = chess.Board()
        # board3d = split_bitboard(boad)
        # print(board3d)

        # unpacked_bits = numpy.unpackbits(binary)
        # print(unpacked_bits)


            
    except Exception as e:
        print(f"An error occurred: {e}")
    # db = EvaluationDataset()
    # # db.delete()
    # db.import_game(".\\ChessML\\Dataset\\lichess_db_standard_rated_2024-02.pgn")
    # db.close()
    # board = chess.Board("6rr/8/8/8/8/8/R7/7R w - - 0 1")
    # print(split_bitboard(board))
    # board.push(chess.Move.from_uci("a2g2"))
    # print(bin(board_to_binary(board))[2::].zfill(64))


if __name__ == "__main__":
    test()
