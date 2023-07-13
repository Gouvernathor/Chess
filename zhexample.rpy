# default hexboard = chess.hex.Board.initial
default hexboard = chess.hex.Board.empty

label game_loop_hex:
    show screen hex_chess_board(hexboard)
    pause
    jump game_loop_hex
