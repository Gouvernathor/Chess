init python:
    import collections

define 1 initial_hex_counter = chess.hex.Board.initial.pieces_counter()
default hexboard = chess.hex.Board.initial
default hex_move_history = []
default hex_eaten = {chess.hex.Color.WHITE: collections.Counter(), chess.hex.Color.BLACK: collections.Counter()}
    # replace in hex_game_loop if using more colors
    # for more colors, will either represent the eaten pieces *of* each color (instead of *by* each color),
    # or will need subcounters i.e red ate 2 blue pawns and 1 yellow pawn
    # and that will require a different registering implementation

define __blue_highlight = dict(
    at=Transform(matrixcolor=TintMatrix("#77f")),
)
define __red_highlight = dict(
    at=Transform(matrixcolor=TintMatrix("#f77")),
)
define __orange_highlight = dict(
    at=Transform(matrixcolor=TintMatrix("#fb7")),
)

label hex_game_loop(
    chosen_move=None,
    button_highlight_properties=collections.defaultdict(dict),
    hexagon_highlight_properties=collections.defaultdict(dict),
    ):

    while not chosen_move:
        if hexboard.is_check(hexboard.active):
            # highlight the king and the checking pieces
            python hide:
                button_highlight_properties[hexboard.king_hex(hexboard.active)] |= __orange_highlight
                for checkermove in hexboard.generate_checkers(hexboard.active):
                    button_highlight_properties[checkermove.from_hex] |= __red_highlight

        call screen hex_chess_display(hexboard, action_function=Return, button_properties_function=button_highlight_properties.get)

        python hide:
            renpy.suspend_rollback(True)

            if isinstance(_return, chess.hex.Move):
                store.chosen_move = _return
                renpy.notify(f"chosen move: {_return}")

            else:
                piece = hexboard.storage[_return]

                button_highlight_properties.clear()
                hexagon_highlight_properties.clear()
                if piece is not None:
                    if piece.color == hexboard.active:
                        renpy.notify("piece of the active color")
                        for move in hexboard.generate_legal_moves(_return):
                            button_highlight_properties[move.to_hex] |= __blue_highlight | {"action":Return(move)}

                    else:
                        renpy.notify("piece not of the active color")
                else:
                    renpy.notify("no piece found at location")

    show screen hex_chess_display(hexboard, sensitive=False)
    $ renpy.suspend_rollback(False)

    if hexboard.is_promotion_hex(chosen_move.to_hex) and hexboard.storage[chosen_move.from_hex].kind == chess.hex.PieceType.PAWN:
        # promotion
        if board.active == chess.hex.Color.WHITE:
            $ renpy.dynamic(symbols="♕♖♗♘")
        else:
            $ renpy.dynamic(symbols="♛♜♝♞")
        play sound promotion
        menu:
            "Your pawn earned a promotion !\nChoose a piece to promote it to."
            "[symbols[0]] Queen [symbols[0]]":
                $ chosen_move = chosen_move.replace(promotion=chess.hex.PieceType.QUEEN)
            "[symbols[1]] Rook [symbols[1]]":
                $ chosen_move = chosen_move.replace(promotion=chess.hex.PieceType.ROOK)
            "[symbols[2]] Bishop [symbols[2]]":
                $ chosen_move = chosen_move.replace(promotion=chess.hex.PieceType.BISHOP)
            "[symbols[3]] Knight [symbols[3]]":
                $ chosen_move = chosen_move.replace(promotion=chess.hex.PieceType.KNIGHT)

    if hexboard.taken_hex(chosen_move) is not None:
        # this is a take
        $ renpy.notify("take !")
        play sound take

    $ hex_move_history.append(hexboard.notation(chosen_move))
    $ hexboard = hexboard.make_move(chosen_move)

    # updating the loss count
    python hide:
        etn = initial_hex_counter - hexboard.pieces_counter()
        for d in hex_eaten.values():
            d.clear()
        for piece in sorted(etn.elements(), reverse=True):
            hex_eaten[piece.color][piece.kind] += 1

    # refresh
    show screen hex_chess_display(hexboard, sensitive=False)

    if hexboard.is_checkmate(hexboard.active):
        jump __checkmate

    elif hexboard.is_stalemate(hexboard.active):
        jump __stalemate

    elif hexboard.is_check(hexboard.active):
        $ renpy.notify("check !")
        play sound check

    jump hex_game_loop

label __checkmate(winner=chess.hex.Color(not hexboard.active), loser=hexboard.active):
    play sound checkmate
    "Checkmate !\n[winner.name!cl] wins, [loser.name!cl] loses."
    return

label __stalemate(color=hexboard.active):
    play sound stalemate
    "Stalemate !\n[color.name!cl] has no available move."
    return

screen hex_chess_display(hexboard, *args, sensitive=True, locations=.3, **kwargs):
    sensitive sensitive
    use hex_chess_board(hexboard, *args, locations=locations, **kwargs)

    vbox:
        at topleft
        text "To play : [hexboard.active.name!t]"
        null height 30
        text "Eaten by black :"
        for piece_kind, count in hex_eaten[chess.hex.Color.WHITE].items():
            hbox:
                for i in range(count):
                    add chess.Piece(piece_kind, color=chess.hex.Color.WHITE).displayable():
                        xysize (52, 52)

    vbox:
        at left
        text "Eaten by white :"
        vbox:
            yminimum 100
            for piece_kind, count in hex_eaten[chess.hex.Color.BLACK].items():
                hbox:
                    for i in range(count):
                        add chess.Piece(piece_kind, color=chess.hex.Color.BLACK).displayable():
                            xysize (52, 52)

    vbox:
        at topright
        text "Move history :"
        null height 30
        for move in reversed(hex_move_history):
            text move
