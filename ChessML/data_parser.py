from io import BytesIO
import chess.pgn
import chess.engine
import numpy
import chess
import pymysql
import sys
import os
import base64
import traceback
import torch
from dotenv import load_dotenv
"""
EvaluationDataset takes single random row from the SQLite table and preprocesses it by extracting the
binary value in raw bytes, converting those bytes to floats using numpy’s frombuffer and unpackbits functions,
and forming the required 808 length float array as input. The evaluation value is is extracted and bounded between -15 and 15.
"""
DEPTH = 16


# ======================AWS RDS MySQL Connection===============================
class EvaluationDataset():

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
        self.query = "SELECT * FROM ChessData"

    def __next__(self):
        try:
            self.cursor = self.db.cursor()
            self.cursor.execute("SELECT bin, eval FROM ChessData ORDER BY RAND() LIMIT 1")
            result = self.cursor.fetchone()
            # convert binary to numpy array
            binary = BytesIO(result[0])
            binary = numpy.frombuffer(binary.getvalue(), dtype=numpy.uint8)
            binary = binary.reshape(14, 8, 8)

            val = min(result[1], 15) # Checkmate score is 10000 so we bound it to 15, otherwise it's too high for the network
            val = max(val, -15) # Checkmate score is -10000 so we bound it to -15, otherwise it's too low for the network
            
            binary = torch.from_numpy(binary).to(torch.float16)
                        
            return binary, val
        except Exception as e:
            print("Database connection failed due to {}".format(e))
            raise
    
    def __getitem__(self, idx):
        try:
            self.cursor = self.db.cursor()
            self.cursor.execute("SELECT bin, eval FROM ChessData WHERE id = %s", (idx + 1,))
            result = self.cursor.fetchone()
            # convert binary to numpy array
            binary = BytesIO(result[0])
            binary = numpy.frombuffer(binary.getvalue(), dtype=numpy.uint8)
            binary = binary.reshape(14, 8, 8)

            val = min(result[1], 15) # Checkmate score is 10000 so we bound it to 15, otherwise it's too high for the network
            val = max(val, -15) # Checkmate score is -10000 so we bound it to -15, otherwise it's too low for the network
            
            binary = torch.from_numpy(binary).to(torch.float16)
            
            return binary, val
        except Exception as e:
            print("Database connection failed due to {}".format(e))
            raise
        
    def __len__(self):
        try:
            self.cursor.execute("SELECT COUNT(*) FROM ChessData")
            count = self.cursor.fetchone()[0]
            return count
        except Exception as e:
            print("Database connection failed due to {}".format(e))
            raise
        
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
            
    def __iter__(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(self.query)
        for row in cursor:
            yield row
  
        
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

                        print("Inserting into database: ", game_id, len(binary), eval)
                        print(board.fen())
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
    

    # Evaluate the board using Stockfish: Positive score means white is winning, negative score means black is winning
    def stock_fish_eval(self, board, depth):
        with chess.engine.SimpleEngine.popen_uci(
            ".\\ChessML\\stockfish\\stockfish-windows-x86-64-avx2.exe"
        ) as sf:
            result = sf.analyse(board, chess.engine.Limit(depth=depth)).get("score")
            print(board)
            return result.white().score(mate_score=10000) / 100
        
        
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
    # 1D array with 14 total, arrays of 64 bits
    # 14 arrays:
    # 6 arrays for white pieces
    # 6 arrays for black pieces
    # 2 arrays add attacks and valid moves so the network knows what is being attacked
    # concatenate the turn, castling rights, and en passant square to the end of the array
    # 903 total bits
    bitboards = numpy.array([], dtype=numpy.uint8)
    
    for piece in chess.PIECE_TYPES:
        bitboard = numpy.zeros(64, dtype=numpy.uint8)
        for square in board.pieces(piece, chess.WHITE):
            bitboard[square] = 1
        bitboards = numpy.append(bitboards, bitboard)
    for piece in chess.PIECE_TYPES:
        bitboard = numpy.zeros(64, dtype=numpy.uint8)
        for square in board.pieces(piece, chess.BLACK):
            bitboard[square] = 1
        bitboards = numpy.append(bitboards, bitboard)
    
    # Add attacks and valid moves
    aux = board.turn
    board.turn = chess.WHITE
    bitboard = numpy.zeros(64, dtype=numpy.uint8)
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        bitboard[i * 8 + j] = 1
    bitboards = numpy.append(bitboards, bitboard)
    
    board.turn = chess.BLACK
    bitboard = numpy.zeros(64, dtype=numpy.uint8)
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        bitboard[i * 8 + j] = 1
    bitboards = numpy.append(bitboards, bitboard)
    board.turn = aux

    # bitboards = numpy.append(bitboards, [board.turn])

    # bitboards = numpy.append(bitboards, [
    #     board.has_kingside_castling_rights(chess.WHITE),
    #     board.has_queenside_castling_rights(chess.WHITE),
    #     board.has_kingside_castling_rights(chess.BLACK),
    #     board.has_queenside_castling_rights(chess.BLACK)
    # ])

    # # Add the check status bits
    # bitboards = numpy.append(bitboards, [
    #     board.is_check(),
    #     board.is_checkmate()
    # ])
    bitboards = bitboards.reshape(14, 8, 8)
    binary = numpy.frombuffer(bitboards, dtype=numpy.uint8)
    return binary.tobytes()




def test():
    try:
        conn = pymysql.connect(
            host="chessai.ci79l2mawwys.us-west-1.rds.amazonaws.com",
            user="admin",
            password="chessengine",
            db="chessai",
        )

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM ChessData")
        count = cur.fetchone()[0]
        print(f"Number of rows in ChessData: {count}")

        # cur.execute("SELECT * FROM ChessData ORDER BY RAND() LIMIT 2")
        # rows = cur.fetchall()


        pass
            
    except Exception as e:
        print(f"An error occurred: {e}")
    db = EvaluationDataset()
    # db.delete()
    db.import_game(".\\ChessML\\Dataset\\lichess_db_standard_rated_2024-02.pgn")





if __name__ == "__main__":
    test()
