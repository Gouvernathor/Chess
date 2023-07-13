default hexboard = chess.hex.Board.initial

label game_loop_hex:
    show screen hex_chess_board(hexboard)
    pause
    jump game_loop_hex
