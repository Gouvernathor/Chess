screen chess_board(board=chess.Board.empty,
    white_board_color=chess.WHITE_BOARD_COLOR, black_board_color=chess.BLACK_BOARD_COLOR
    ):
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
                    add piece.displayable() align (.5, .5) fit "contain"
