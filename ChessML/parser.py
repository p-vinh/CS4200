import chess.pgn
import chess.engine
import boto3
import numpy
import chess
import pymysql
import sys
import os

"""
EvaluationDataset takes single random row from the SQLite table and preprocesses it by extracting the
binary value in raw bytes, converting those bytes to floats using numpy’s frombuffer and unpackbits functions,
and forming the required 808 length float array as input. The evaluation value is is extracted and bounded between -15 and 15.
"""
DEPTH = 16

# ======================AWS RDS MySQL Connection===============================
class EvaluationDataset():
    # ENDPOINT = "mysqldb.123456789012.us-east-1.rds.amazonaws.com"
    # PORT = "3306"
    # USER = "jane_doe"
    # REGION = "us-west-1"
    # DBNAME = "mydb"
    def __init__(self, endpoint, port, user, region, dbname):
        self.endpoint = endpoint
        self.port = port
        self.user = user
        self.dbname = dbname
        self.token = client.generate_db_auth_token(
            DBHostname=endpoint, Port=port, DBUsername=user, Region=region
        )
        os.environ["LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN"] = "1"
        # gets the credentials from .aws/credentials
        session = boto3.Session(profile_name="default")
        client = session.client("rds")

    def connect(self):
        try:
            conn = pymysql.connect(
                host=self.endpoint,
                user=self.user,
                passwd=self.token,
                port=self.port,
                database=self.dbname,
                ssl_ca="SSLCERTIFICATE",
            )
            return conn
        except Exception as e:
            print("Database connection failed due to {}".format(e))

    def import_game(self):
        with open(".\\Dataset\\lichess_db_standard_rated_2024-02.pgn") as pgn:
            game = chess.pgn.read_game(pgn)

        try:
            conn = self.connect()
            cur = conn.cursor()
            game_id = 1
            while game is not None:
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    eval = self.stock_fish_eval(board, DEPTH)
                    binary = format(eval, ".2f")

                    cur.execute(
                        "INSERT INTO your_table (id, fen, binary, eval) VALUES (%s, %s, %s, %s)",
                        (game_id, board.fen(), binary, eval),
                    )
                    game_id += 1
                game = chess.pgn.read_game(pgn)
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print("Database connection failed due to {}".format(e))

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
            
            return {'binary': bin, 'eval': ev}
        
        except Exception as e:
            print("Database connection failed due to {}".format(e))
            
    # Evaluate the board using Stockfish: Positive score means white is winning, negative score means black is winning
    def stock_fish_eval(self, board, depth):
        with chess.engine.SimpleEngine.popen_uci(
            ".\\stockfish\\stockfish-windows-x86-64-avx2.exe"
        ) as sf:
            result = sf.analyse(board, chess.engine.Limit(depth=depth))
            score = result["score"].white().score()
            return score

# ==========================================================================