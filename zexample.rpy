init python:
    import collections

define initial_counter = chess.Board.initial.pieces_counter()
default board = chess.Board.initial
default move_history = []
default eaten = {chess.BLACK: collections.Counter(), chess.WHITE: collections.Counter()}

define blue_highlight = dict(
    foreground="#77ffff77",
)
define red_highlight = dict(
    background="#ff7777aa",
)
define orange_highlight = dict(
    background="#ffbb77aa",
)

label game_loop(chosen_move=None, highlight_properties=collections.defaultdict(dict)):
    # actions will be reset at each loop, but that's ok
    # it will also be deleted when returning

    while not chosen_move:
        if board.is_check(board.active):
            # highlight the king and the checking pieces
            python hide:
                store.highlight_properties[board.king_square(board.active)] |= orange_highlight
                for checkermove in board.generate_checkers(board.active):
                    store.highlight_properties[checkermove.from_square] |= red_highlight

        call screen main_chess_display(board, action_function=Return, properties_function=highlight_properties.get)

        python hide:
            renpy.suspend_rollback(True)
            # avoid rolling back between highlight changes
            # but still save the first call screen as a checkpoint

            if isinstance(_return, chess.Move):
                store.chosen_move = _return
                renpy.notify(f"registering move {_return}")

            else:
                piece = board.flat_placement[_return]

                store.highlight_properties.clear()
                if piece is not None:
                    if piece.color == board.active:
                        renpy.notify("piece of the active color")
                        for move in board.generate_legal_moves(_return):
                            store.highlight_properties[move.to_square] |= blue_highlight
                            store.highlight_properties[move.to_square]["action"] = Return(move)

                    else:
                        renpy.notify("piece not of the active color")
                else:
                    renpy.notify("no piece found at location")

    show screen main_chess_display(board, sensitive=False)
    # keep it visible
    $ renpy.suspend_rollback(False)
    # allow rolling back again

    if chosen_move.to_square.rankidx() in (0, 7) and board.flat_placement[chosen_move.from_square].kind == chess.PAWN:
        # promotion
        if board.active == chess.WHITE:
            # I know it's inverted, it's because the font color is black so this renders better
            $ renpy.dynamic(symbols="♕♖♗♘")
        else:
            $ renpy.dynamic(symbols="♛♜♝♞")
        play sound promotion
        menu:
            "Your pawn earned a promotion !\nChoose a piece to promote it to."
            "[symbols[0]] Queen [symbols[0]]":
                $ chosen_move = chosen_move.replace(promotion=chess.QUEEN)
            "[symbols[1]] Rook [symbols[1]]":
                $ chosen_move = chosen_move.replace(promotion=chess.ROOK)
            "[symbols[2]] Bishop [symbols[2]]":
                $ chosen_move = chosen_move.replace(promotion=chess.BISHOP)
            "[symbols[3]] Knight [symbols[3]]":
                $ chosen_move = chosen_move.replace(promotion=chess.KNIGHT)

    if board.taken_square(chosen_move) is not None:
        # this is a take
        $ renpy.notify("take !")
        play sound take

    # when a move has been made
    # can't save the move itself for later short of saving the whole board,
    # because the algebraic notation requires the board it was applied to
    $ move_history.append(board.algebraic_notation(chosen_move))
    $ board = board.make_move(chosen_move)

    # updating loss counts
    python hide:
        etn = initial_counter - board.pieces_counter()
        eaten[chess.BLACK].clear()
        eaten[chess.WHITE].clear()
        for piece in sorted(etn.elements(), reverse=True):
            eaten[piece.color][piece.kind] += 1

    # refresh
    show screen main_chess_display(board, sensitive=False)

    if board.is_checkmate(board.active):
        jump checkmate

    elif board.is_stalemate(board.active):
        jump stalemate

    elif board.is_check(board.active):
        $ renpy.notify("check !")
        play sound check

    jump game_loop

label checkmate(winner=chess.Color(not board.active)):
    play sound checkmate
    "Checkmate !\n[winner.name!cl] wins, [board.active.name!cl] loses."
    return

label stalemate:
    play sound stalemate
    "Stalemate : [board.active.name!cl] has no available move."
    return

screen main_chess_display(board, *args, sensitive=True, **kwargs):
    sensitive sensitive
    use chess_board(board, *args, **kwargs)
    vbox: # top-left side
        at topleft
        text "To play : [board.active.name!t]"
        null height 30
        text "Eaten by black :"
        for piece_kind, count in eaten[chess.WHITE].items():
            hbox:
                for i in range(count):
                    add chess.Piece(piece_kind, color=chess.WHITE).displayable():
                        xysize (52, 52)

    vbox: # bottom-left
        at left
        text "Eaten by white :"
        vbox:
            yminimum 100
            for piece_kind, count in eaten[chess.BLACK].items():
                hbox:
                    for i in range(count):
                        add chess.Piece(piece_kind, color=chess.BLACK).displayable():
                            xysize (52, 52)

    vbox: # right
        at topright
        text "Move history :"
        null height 30
        for move in reversed(move_history): # add viewport
            text move
