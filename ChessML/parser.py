import chess.pgn
import chess
import json


def stock_fish_eval(board, depth):
    with chess.engine.SimpleEngine.popen_uci("stockfish") as sf:
        info = sf.analyse(board, chess.engine.Limit(depth=depth))
        return info["score"].relative.score(mate_score=10000)
    
pgn = open(".\\Dataset\\lichess_db_standard_rated_2013-01.pgn")
first_game = chess.pgn.read_game(pgn)

print(first_game.headers)
board = chess.Board()
moves = iter(first_game.mainline_moves())
next_move = next(moves, None)

while next_move is not None:
    print(next_move)
    
    board.push(next_move)
    stock_fish_eval(board, 10)
    next_move = next(moves, None)
    
    


    
# import chess.pgn
# import psycopg2

# # Open the PGN file
# with open('games.pgn') as pgn:
#     game = chess.pgn.read_game(pgn)

# # Connect to the AWS RDS instance
# conn = psycopg2.connect(
#     dbname='your_dbname',
#     user='your_username',
#     password='your_password',
#     host='your_host',
#     port='your_port'
# )
# cur = conn.cursor()

# # Initialize the game id
# game_id = 1

# # Parse the games
# while game is not None:
#     board = game.board()
#     for move in game.mainline_moves():
#         board.push(move)
        
#         # Here you need to replace 'your_eval_function' with the function you use to evaluate the position
#         eval = your_eval_function(board)
        
#         # Convert the eval to a binary format
#         binary = format(eval, '.2f')
        
#         # Insert the data into the database
#         cur.execute(
#             "INSERT INTO your_table (id, fen, binary, eval) VALUES (%s, %s, %s, %s)",
#             (game_id, board.fen(), binary, eval)
#         )
        
#         game_id += 1

#     game = chess.pgn.read_game(pgn)

# # Commit the changes and close the connection
# conn.commit()
# cur.close()
# conn.close()