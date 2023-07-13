# board must be a chess.Board object
# the colors are the colors of the squares
# action_function is a function that takes a square as argument and returns an action, or None
# it can be passed a dict.get bound method, for example
# it can also be passed Return, or functools.partial(SetScreenVariable, "clickedonsquare"), or functools.partial(functools.partial(renpy.notify))
# properties_function is a function that takes a square as argument and returns a dict of properties to be passed to the buttons, or None
# the dicts' keys and values must be valid style property names and values taken by buttons
# the dict may contain the "action" key, in which case it will override the action_function's result (it will still be called though)
screen chess_board(board=chess.Board.empty,
    white_board_color=chess.WHITE_BOARD_COLOR, black_board_color=chess.BLACK_BOARD_COLOR,
    action_function={}.get,
    properties_function={}.get,
    ):
    grid 8 8:
        align (.5, .5)
        at transform:
            fit "contain"
        for square, rownbr, colnbr, piece in board.enumerate():
            fixed:
                xysize (100, 100) # arbitrary, will be scaled by fit contain
                if (rownbr+colnbr)%2:
                    add white_board_color
                else:
                    add black_board_color

                button:
                    xysize (1., 1.)
                    style "default"
                    if piece is not None:
                        add piece.displayable() align (.5, .5) fit "contain"
                    action action_function(square)
                    properties properties_function(square)

init python:
    from math import sqrt
    sqrt3 = sqrt(3)

    class Hexagon(renpy.Displayable):
        def __init__(self, color, linewidth=0, regular=False, side=None, flat_top=True, **properties):
            """
            Regular imposes a regular hexagon within the available space.
            Side imposes a regular hexagon with the given side length, overriding regular.
            """
            # TODO: merge the regular and side attributes but keep the two parameters
            properties.setdefault("subpixel", True)
            super().__init__(**properties)
            self.color = color
            self.linewidth = linewidth
            self.regular = regular
            self.side = side
            self.flat_top = flat_top

        # TODO: add a static method passed a canvas and some sizes and writes the thing on it
        # so that the process can be scaled
        def render(self, width, height, st, at):
            if self.side is not None:
                height = 2*self.side
                width = sqrt3*self.side

            if self.flat_top:
                rv = renpy.Render(height, width)
            else:
                rv = renpy.Render(width, height)

            if self.regular:
                factor = sqrt3/2
                if width > factor*height:
                    newidth = factor*height
                    newheight = height
                else:
                    newidth = width
                    newheight = width/factor
                subx = (width-newidth)/2
                suby = (height-newheight)/2
                if self.flat_top:
                    sub = rv.subsurface((sub7, subx, newheight, newidth))
                else:
                    sub = rv.subsurface((subx, suby, newidth, newheight))
                cv = sub.canvas()
            else:
                newidth = width
                newheight = height
                cv = rv.canvas()

            top_point = (newidth/2, 0)
            bottom_point = (newidth/2, newheight)

            a = -3
            b = 4*newheight
            c = newidth**2/4 - newheight**2
            delta = b**2-4*a*c
            x1 = (-b+sqrt(delta))/(2*a)
            x2 = (-b-sqrt(delta))/(2*a)
            if 0 < x1 < newheight/2:
                x = x1
            else:
                x = x2

            topleft_point = (0, x)
            bottomleft_point = (0, newheight-x)
            topright_point = (newidth, x)
            bottomright_point = (newidth, newheight-x)

            pointslist = [
                top_point,
                topright_point,
                bottomright_point,
                bottom_point,
                bottomleft_point,
                topleft_point,
            ]

            if self.flat_top:
                pointslist = [(y, x) for (x, y) in pointslist]

            cv.polygon(
                self.color,
                pointslist,
                self.linewidth,
            )

            if self.regular:
                if self.flat_top:
                    rv.blit(sub, (suby, subx))
                else:
                    rv.blit(sub, (subx, suby))

            return rv

screen hex_chess_board(board=chess.hex.Board.empty,
    white_board_color=chess.hex.WHITE_BOARD_COLOR, grey_board_color=chess.hex.GREY_BOARD_COLOR, black_board_color=chess.hex.BLACK_BOARD_COLOR,
    action_function={}.get,
    properties_function={}.get,
    ):
    default side = absolute(100)
    fixed:
        align (.5, .5)
        # just for the ratio, will be scaled by fit contain
        # `side` being the side length of a hex if it is regular
        # (which is assumed in the height calculation)
        # which also means half of its width when its points are to the side
        xysize (10*(3/2)*side+2*side, 11*sqrt3*side)
        at transform:
            fit "contain"

        for hex in chess.hex.Hex.range():
            $ piece = board.storage[hex]
            button:
                style "default"
                xpos side*3/2*hex.q
                ypos side*sqrt3*((hex.q-5)/2+hex.r)
                action action_function(hex)
                focus_mask True
                fixed: # needs to be explicit otherwise fit_first is not accepted
                    fit_first True
                    add Hexagon(
                        color=(black_board_color, grey_board_color, white_board_color)[(hex.q-hex.r+1)%3],
                        side=side,
                    )
                    fixed:
                        xysize (side*1.25, side*1.25)
                        align (.5, .5)
                        if piece is not None:
                            add piece.displayable() align (.5, .5) fit "contain"
                    text "[hex.q], [hex.r], [hex.s]":
                        align (.5, .5)
                        size 20
                        textalign .5
