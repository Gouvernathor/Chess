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

    class Hexagon(renpy.Displayable):
        def __init__(self, color, linewidth=0, regular=False, side=None, **properties):
            """
            Regular imposes a regular hexagon within the available space.
            Side imposes a regular hexagon with the given side length, overriding regular.
            """
            # TODO: merge the two attributes but keep the two parameters
            super().__init__(**properties)
            self.color = color
            self.linewidth = linewidth
            self.regular = regular
            self.side = side

        # TODO: add a static method passed a canvas and some sizes and writes the thing on it
        # so that the process can be scaled
        def render(self, width, height, st, at):
            if self.side is not None:
                height = 2*self.side
                width = sqrt(3)*self.side

            rv = renpy.Render(width, height)

            if self.regular:
                factor = sqrt(3)/2
                if width > factor*height:
                    newidth = factor*height
                    newheight = height
                else:
                    newidth = width
                    newheight = width/factor
                subx = (width-newidth)/2
                suby = (height-newheight)/2
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
            print(x1, x2, newidth, newheight)
            if 0 < x1 < newheight/2:
                x = x1
            else:
                x = x2

            topleft_point = (0, x)
            bottomleft_point = (0, newheight-x)
            topright_point = (newidth, x)
            bottomright_point = (newidth, newheight-x)

            cv.polygon(
                self.color,
                [
                    top_point,
                    topright_point,
                    bottomright_point,
                    bottom_point,
                    bottomleft_point,
                    topleft_point,
                ],
                self.linewidth,
            )

            if self.regular:
                rv.blit(sub, (subx, suby))

            return rv
