import chess.pgn
import chess.engine
import boto3
import numpy
import chess
import pymysql
import sys
import base64
import os
import traceback
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

        self.square_to_index = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
        }

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
            self.cursor.close()
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
                        binary = self.board_to_binary(board)
                        
                        print("Inserting into database: ", game_id, eval)     
                        if eval is not None:             
                            self.cursor.execute(
                                "INSERT INTO ChessData (id, fen, bin, eval) VALUES (%s, %s, %s, %s)",
                                (game_id, board.fen(), binary.tobytes(), eval),
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
            ".\\stockfish\\stockfish-windows-x86-64-avx2.exe"
        ) as sf:
            result = sf.analyse(board, chess.engine.Limit(depth=depth))
            print(board)
            
            score = result["score"].white().score()
            return score / 100.0 if score is not None else None

    def board_to_binary(self, board):
        binary = numpy.zeros(64, dtype=numpy.uint8)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                index = self.square_to_index[chess.square_name(square)[0]]
                binary[index] = piece.piece_type
        return binary

    def square_to_coordinates(self, square):
        letter = chess.square_name(square)
        return 8 - int(letter[1]), self.square_to_index[letter[0]]

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


def test():
    conn = pymysql.connect(
        host="chessai.ci79l2mawwys.us-west-1.rds.amazonaws.com",
        user="admin",
        password="chessengine",
        db="chessai",
    )
    
    cur = conn.cursor()
    cur.execute("SELECT * FROM ChessData ORDER BY RAND() LIMIT 10")
    rows = cur.fetchall()
    
    for row in rows:
        print(row)
    # db = EvaluationDataset()
    # db.delete()
    # db.import_game(".\\Dataset\\lichess_db_standard_rated_2024-02.pgn")
    # db.close()


if __name__ == "__main__":
    test()
