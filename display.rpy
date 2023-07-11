screen chess_board(fen=chess.START_FEN, white_board_color=chess.WHITE_BOARD_COLOR, black_board_color=chess.BLACK_BOARD_COLOR):
    python:
        if fen:
            board = chess.Board.from_fen(fen)
        else:
            board = chess.Board.empty
    grid 8 8:
        xysize (1., 1.)
        align (.5, .5)
        at transform:
            fit "contain"
        for square, rownbr, colnbr, piece in board.enumerate():
            button:
                style "default"
                xysize (100, 100) # arbitrary, will be scaled by fit contain
                if (rownbr+colnbr)%2:
                    add white_board_color
                else:
                    add black_board_color
                if piece is not None:
                    add piece.displayable() align (.5, .5)
