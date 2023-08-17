import renpy # type: ignore
python_object = object
"""renpy
init python in chess:
"""
_constant = True

import collections
import dataclasses
import enum
import functools

class Color(enum.IntEnum):
    BLACK = False
    WHITE = True
BLACK, WHITE = Color

WHITE_BOARD_COLOR = "#f0d9b5"
BLACK_BOARD_COLOR = "#b58863"

@functools.total_ordering
class PieceType(enum.Enum):
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = "p", "n", "b", "r", "q", "k"
    def __init__(self, value):
        self.cmp_value = len(type(self).__members__) + 1
    def __lt__(self, other):
        if type(self) is type(other):
            return self.cmp_value < other.cmp_value
        return NotImplemented
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = PieceType

FILE_NAMES = ("a", "b", "c", "d", "e", "f", "g", "h")
RANK_NAMES = ("1", "2", "3", "4", "5", "6", "7", "8")
class Square(enum.IntEnum):
    (
        A1, B1, C1, D1, E1, F1, G1, H1,
        A2, B2, C2, D2, E2, F2, G2, H2,
        A3, B3, C3, D3, E3, F3, G3, H3,
        A4, B4, C4, D4, E4, F4, G4, H4,
        A5, B5, C5, D5, E5, F5, G5, H5,
        A6, B6, C6, D6, E6, F6, G6, H6,
        A7, B7, C7, D7, E7, F7, G7, H7,
        A8, B8, C8, D8, E8, F8, G8, H8,
    ) = range(64)

    # get the internal index for the given square ("a1" -> 0) : Square.NAMES.index()
    # get the square name from the internal index (0 -> "a1") : Square.NAMES[]
    @classmethod
    def square(cls, fileidx: int, rankidx: int):
        """
        takes 0-based indexes, because 1-based rank index is a string
        """
        return cls(rankidx*8 + fileidx)
    def fileidx(self):
        return self & 7 # faster than square%8
    def rankidx(self):
        return self >> 3 # faster than square//8
    def indexes(self):
        """
        returns file, rank
        """
        r, f = divmod(self, 8)
        return f, r

    @classmethod
    def range(cls, start=64, *args):
        for i in range(start, *args):
            yield cls(i)

    def __add__(self, other):
        if isinstance(other, Square):
            return NotImplemented
        rv = super().__add__(other)
        if rv is NotImplemented:
            return NotImplemented
        return type(self)(rv)
    __radd__ = __add__
    def __sub__(self, other):
        if isinstance(other, Square):
            return NotImplemented
        rv = super().__sub__(other)
        if rv is NotImplemented:
            return NotImplemented
        return type(self)(rv)
    # rsub doesn't make sense, just like neg

UNICODE_PIECE_SYMBOLS = {
    "R": "♖", "r": "♜",
    "N": "♘", "n": "♞",
    "B": "♗", "b": "♝",
    "Q": "♕", "q": "♛",
    "K": "♔", "k": "♚",
    "P": "♙", "p": "♟",
}
@dataclasses.dataclass(frozen=True, order=True)
class Piece(python_object):
    kind: PieceType
    color: Color
    def symbol(self) -> str:
        rv = self.kind.value
        if self.color == BLACK:
            return rv.lower()
        elif self.color == WHITE:
            return rv.upper()
        raise ValueError("Invalid color")
    def unicode_symbol(self):
        return UNICODE_PIECE_SYMBOLS[self.symbol()]
    @classmethod
    def from_symbol(cls, symbol):
        return cls(PieceType(symbol.lower()), Color(symbol.isupper()))

    def displayable(self):
        if self.color == WHITE:
            rv = "w"
        elif self.color == BLACK:
            rv = "b"
        else:
            return None
        return rv + self.kind.value

@dataclasses.dataclass(frozen=True)
class Move(python_object):
    from_square: Square
    to_square: Square
    promotion: PieceType|None = None
    def __str__(self):
        rv = self.from_square.name.lower() + self.to_square.name.lower()
        if self.promotion:
            rv += self.promotion.value
        return rv
    replace = dataclasses.replace

    def reverse(self):
        """
        Warning : reverses of a promotion move are not valid moves.
        Reversing those again will not give the original move.

        Otherwise, the reverse of the reverse is the original move.
        """
        if self.promotion is not None:
            return Move(self.to_square, self.from_square, PAWN)
        return Move(self.to_square, self.from_square)

    def diff_tuple(self):
        """
        returns file diff, rank diff
        """
        fromfile, fromrank = self.from_square.indexes()
        tofile, torank = self.to_square.indexes()
        return tofile-fromfile, torank-fromrank

class Castling(collections.namedtuple("Castling", ("white_kingside", "white_queenside", "black_kingside", "black_queenside"))):
    __slots__ = ()

    white_kingside: bool
    white_queenside: bool
    black_kingside: bool
    black_queenside: bool

    @classmethod
    def from_fen(cls, fen):
        """
        not a true fen, just the castling part
        """
        return cls("K" in fen, "Q" in fen, "k" in fen, "q" in fen)

    def to_fen(self):
        rv = ""
        if self.white_kingside:
            rv += "K"
        if self.white_queenside:
            rv += "Q"
        if self.black_kingside:
            rv += "k"
        if self.black_queenside:
            rv += "q"
        if not rv:
            rv = "-"
        return rv

    __str__ = to_fen

@dataclasses.dataclass(frozen=True)
class Board(python_object):
    """
    contains the same information as a FEN
    """

    flat_placement: tuple[Piece|None, ...] # type: tuple[((Piece|None),)*64]
    active: Color|None
    castling: Castling
    enpassant: Square|None # aka ep_square
    halfclock: int = 0
    fullclock: int = 1

    @classmethod
    def from_fen(cls, fen):
        """
        taking a full fen
        """
        placement, active, castling, enpassant, halfclock, fullclock = fen.split(" ")

        ranks = placement.split("/")
        flat_placement = []
        for rank in reversed(ranks):
            for char in rank:
                if char.isdigit():
                    flat_placement.extend([None]*int(char))
                else:
                    flat_placement.append(Piece.from_symbol(char))

        if active == "w":
            active = WHITE
        elif active == "b":
            active = BLACK
        else:
            active = None

        castling = Castling.from_fen(castling)

        if enpassant == "-":
            enpassant = None
        else:
            enpassant = Square(enpassant)

        return cls(tuple(flat_placement), active, castling, enpassant, int(halfclock), int(fullclock))

    def board_fen(self):
        ranks = []
        for rankidx in reversed(range(8)):
            rank = []
            for piece in self.flat_placement[rankidx*8:rankidx*8+8]:
                if piece is None:
                    if rank and isinstance(rank[-1], int):
                        rank[-1] += 1
                    else:
                        rank.append(1)
                else:
                    rank.append(piece.symbol())
            ranks.append("".join(map(str, rank)))
        return "/".join(ranks)

    def to_fen(self):
        raise NotImplementedError

    __str__ = to_fen

    def __getitem__(self, square: tuple[int|str, int|str]):
        """
        takes a tuple of ints (fileidx 0-7, rankidx 0-7),
        or strs (filename "a"-"h", rankname "1"-"8")
        """
        f, r = square
        if isinstance(f, str):
            f = FILE_NAMES.index(f)
        if isinstance(r, str):
            r = RANK_NAMES.index(r)

        return self.flat_placement[Square.square(f, r)]

    # to get the type/color somewhere, board[square].kind/color

    def king_square(self, color: Color):
        found = Piece(KING, color)
        for sqn, piece in enumerate(self.flat_placement):
            if piece == found:
                return Square(sqn)
        raise ValueError("King not found")

    def pieces_counter(self):
        rv = collections.Counter(self.flat_placement)
        del rv[None]
        return rv

    def is_zeroing(self, move: Move):
        """
        returns true if the move is a capture or a pawn move
        the move must be valid/legal
        """
        if self.flat_placement[move.from_square].kind == PAWN: # type: ignore
            return True
        if self.flat_placement[move.to_square] is not None:
            return True
        return False

    def taken_square(self, move: Move):
        """
        Returns a square this move would take, or None.
        Supports invalid moves where no piece is present to make it.
        Does not check for invalidity when moving and target pieces are of the same color.
        """
        # simple case
        to_square = move.to_square
        if self.flat_placement[to_square] is not None:
            return to_square

        # enpassant
        from_square = move.from_square
        attacker = self.flat_placement[from_square]
        if attacker is not None and attacker.kind == PAWN:
            move_filediff = move.diff_tuple()[0]
            if move_filediff and (to_square == self.enpassant):
                return Square.square(to_square.fileidx(), from_square.rankidx())

        return None

    def generate_attackers(self, square: Square, color: Color):
        """
        generates moves by the opposite color which attack the given square
        moves that would put or leave the king of the opposite color in check are allowed (per the rules)
        generates enpassant attacks, even though those would never put a king in check
        """
        for it_square in Square.range():
            piece = self.flat_placement[it_square]
            if (piece is None) or (piece.color == color):
                continue
            for move in self.generate_moves(it_square, castling=False):
            # castling = False because a castle move could never kill : the destination squares for both the king and the rook must be empty
                if self.taken_square(move) == square and self.is_legal(move, check_check=False, check_turn=False):
                    yield move
                    break

    def is_under_attack(self, square: Square, color: Color):
        """
        returns true if the given square is attacked by the opposite color
        """
        try:
            next(self.generate_attackers(square, color))
        except StopIteration:
            return False
        else:
            return True

    def generate_checkers(self, color: Color):
        """
        generates moves by which the king of the given color is attacked
        moves that would put or leave the king of the opposite color in check are allowed (per the rules)
        """
        king_square = self.king_square(color)
        yield from self.generate_attackers(king_square, color)

    def is_check(self, color: Color|None = None):
        """
        returns true if the given color is in check
        """
        if color is None:
            return self.is_check(WHITE) or self.is_check(BLACK)

        try:
            next(self.generate_checkers(color))
        except StopIteration:
            return False
        else:
            return True

    def is_stalemate(self, color: Color|None = None):
        """
        does not check for check
        """
        if color is None:
            return self.is_stalemate(WHITE) or self.is_stalemate(BLACK)

        for sqn, piece in enumerate(self.flat_placement):
            square = Square(sqn)
            if piece is not None and piece.color == color:
                for move in self.generate_legal_moves(square):
                    if not self.make_move(move).is_check(color):
                        return False

        return True

    def is_checkmate(self, color: Color|None = None):
        return self.is_check(color) and self.is_stalemate(color)

    def generate_castling_moves(self, *, color: Color|None = None, square: Square|None = None):
        """
        generates all possible castling moves on the current board, of the given color if provided and from the given square if provided
        does not check if the moves end up in self-check
        """
        castling = self.castling
        if (color in (WHITE, None)) and (square in (Square.E1, None)):
            if castling.white_kingside:
                if self.flat_placement[Square.G1] is None and self.flat_placement[Square.F1] is None:
                    yield Move(Square.E1, Square.G1)
            if castling.white_queenside:
                if self.flat_placement[Square.D1] is None and self.flat_placement[Square.C1] is None and self.flat_placement[Square.B1] is None:
                    yield Move(Square.E1, Square.C1)
        if (color in (BLACK, None)) and (square in (Square.E8, None)):
            if castling.black_kingside:
                if self.flat_placement[Square.G8] is None and self.flat_placement[Square.F8] is None:
                    yield Move(Square.E8, Square.G8)
            if castling.black_queenside:
                if self.flat_placement[Square.D8] is None and self.flat_placement[Square.C8] is None and self.flat_placement[Square.B8] is None:
                    yield Move(Square.E8, Square.C8)

    def generate_moves(self, square: Square|None = None, castling=True):
        """
        generates all possible moves from the given location on the current board
        if not given a starting square, generates all possible moves of the current player
        will only generate moves without promotion
        does not check if the moves end up in self-check
        """
        if square is None:
            for square in Square.range():
                piece = self.flat_placement[square]
                if (piece is not None) and (piece.color == self.active):
                    yield from self.generate_moves(square)
            return

        piece = self.flat_placement[square]
        if piece is None:
            return
        kind = piece.kind
        color = piece.color
        fileidx, rankidx = square.indexes()

        if kind == PAWN:
            if rankidx == (7 if color == WHITE else 0):
                # not supposed to happen per official rules bc promotion is mandatory to a non-pawn kind
                # but you never know who might program their game badly
                return

            enpassant = self.enpassant
            direction = 1 if color == WHITE else -1

            # front
            if self.flat_placement[square+8*direction] is None:
                yield Move(square, square+8*direction)

                # front 2
                if rankidx == (1 if color == WHITE else 6) and self.flat_placement[square+16*direction] is None:
                    yield Move(square, square+16*direction)

            if color == WHITE:
                # capture left
                if fileidx != 0:
                    destination = square+7
                    target = self.flat_placement[destination]
                    if (enpassant == destination) or (target is not None) and (target.color == BLACK):
                        yield Move(square, destination)
                # capture right
                if fileidx != 7:
                    destination = square+9
                    target = self.flat_placement[destination]
                    if (enpassant == destination) or (target is not None) and (target.color == BLACK):
                        yield Move(square, destination)
            else:
                # capture left
                if fileidx != 0:
                    destination = square-9
                    target = self.flat_placement[destination]
                    if (enpassant == destination) or (target is not None) and (target.color == WHITE):
                        yield Move(square, destination)
                # capture right
                if fileidx != 7:
                    destination = square-7
                    target = self.flat_placement[destination]
                    if (enpassant == destination) or (target is not None) and (target.color == WHITE):
                        yield Move(square, destination)

        elif kind == KING:
            if castling:
                castling_moves = set(self.generate_castling_moves(color=color, square=square))
                if castling_moves and not self.is_check(color):
                    yield from castling_moves

            for rankdiff in (-1, 0, 1):
                if rankidx == 0 and rankdiff == -1:
                    continue
                if rankidx == 7 and rankdiff == 1:
                    continue
                for filediff in (-1, 0, 1):
                    if fileidx == 0 and filediff == -1:
                        continue
                    if fileidx == 7 and filediff == 1:
                        continue
                    target = self[fileidx+filediff, rankidx+rankdiff]
                    if target is None or target.color != color:
                        yield Move(square, square+rankdiff*8+filediff)

        if kind in (ROOK, QUEEN):
            if rankidx != 0:
                for destidx in range(rankidx-1, -1, -1):
                    destination = Square.square(fileidx, destidx)
                    target = self.flat_placement[destination]
                    if target is None:
                        yield Move(square, destination)
                    elif target.color != color:
                        yield Move(square, destination)
                        break
                    else:
                        break
            if rankidx != 7:
                for destidx in range(rankidx+1, 8):
                    destination = Square.square(fileidx, destidx)
                    target = self.flat_placement[destination]
                    if target is None:
                        yield Move(square, destination)
                    elif target.color != color:
                        yield Move(square, destination)
                        break
                    else:
                        break
            if fileidx != 0:
                for destidx in range(fileidx-1, -1, -1):
                    destination = Square.square(destidx, rankidx)
                    target = self.flat_placement[destination]
                    if target is None:
                        yield Move(square, destination)
                    elif target.color != color:
                        yield Move(square, destination)
                        break
                    else:
                        break
            if fileidx != 7:
                for destidx in range(fileidx+1, 8):
                    destination = Square.square(destidx, rankidx)
                    target = self.flat_placement[destination]
                    if target is None:
                        yield Move(square, destination)
                    elif target.color != color:
                        yield Move(square, destination)
                        break
                    else:
                        break

        if kind in (BISHOP, QUEEN):
            for rankdiff in (-1, 1):
                for filediff in (-1, 1):
                    for diffmagn in range(1, 8):
                        if rankidx+rankdiff*diffmagn not in range(8):
                            break
                        if fileidx+filediff*diffmagn not in range(8):
                            break
                        destination = Square.square(fileidx+filediff*diffmagn, rankidx+rankdiff*diffmagn)
                        target = self.flat_placement[destination]
                        if target is None:
                            yield Move(square, destination)
                        elif target.color != color:
                            yield Move(square, destination)
                            break
                        else:
                            break

        if kind == KNIGHT:
            for filediff, rankdiff in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
                if rankidx+rankdiff not in range(8):
                    continue
                if fileidx+filediff not in range(8):
                    continue
                destination = Square.square(fileidx+filediff, rankidx+rankdiff)
                target = self.flat_placement[destination]
                if (target is None) or (target.color != color):
                    yield Move(square, destination)

    def is_legal(self, move: Move, check_check=True, check_turn=True):
        """
        returns true if the move is legal
        """
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion

        # check the moving piece
        piece = self.flat_placement[from_square]
        if piece is None:
            return False

        # check the turn
        if check_turn and (piece.color != self.active):
            return False

        # check promotion
        if promotion is not None:
            if piece.kind != PAWN:
                return False
            if piece.color == WHITE and to_square.rankidx() != 7:
                return False
            if piece.color == BLACK and to_square.rankidx() != 0:
                return False
            if promotion in (PAWN, KING):
                return False

        # check the destination
        dest_piece = self.flat_placement[to_square]
        if dest_piece is not None:
            if dest_piece.color == piece.color:
                return False

        filediff, rankdiff = move.diff_tuple()

        # castling
        # TODO: test if the intermediate square, aka the rook's destination, is under attack
        if (piece.kind == KING) and not (check_check and self.is_check(piece.color)) and (abs(filediff) > 1 or abs(rankdiff) > 1):
            # so this is a castling move
            # test if we would generate it
            if move not in self.generate_castling_moves(color=piece.color):
                return False
            # but still leave room for the check check

        # kind-dependant check
        # only if it's not a castling move
        elif move.replace(promotion=None) not in self.generate_moves(from_square, castling=False):
            return False

        if check_check:
            # check if the move puts our own king in check
            if self.make_move(move).is_check(piece.color):
                return False

        return True

    def generate_legal_moves(self, square: Square|None = None):
        for move in self.generate_moves(square):
            if self.is_legal(move):
                yield move

    def make_move(self, move: Move):
        """
        aka push
        does not return which piece was captured, it's the caller's responsibility to compare board.pieces_counter before and after
        """
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion
        filediff, rankdiff = move.diff_tuple()

        enpassant = None
        halfclock = self.halfclock + 1
        fullclock = self.fullclock
        if self.active == BLACK:
            fullclock += 1
        flat_placement = list(self.flat_placement)

        if self.is_zeroing(move):
            halfclock = 0

        piece = flat_placement[from_square]
        if piece is None:
            raise ValueError("No piece on square, invalid move")
        flat_placement[from_square] = None

        # update castling rights
        castling = self.castling
        if piece.kind == KING:
            if piece.color == WHITE:
                castling = castling._replace(white_kingside=False, white_queenside=False)
            elif piece.color == BLACK:
                castling = castling._replace(black_kingside=False, black_queenside=False)
        elif piece.kind == ROOK:
            if piece.color == WHITE:
                if from_square == Square.A1: # long castling
                    castling = castling._replace(white_queenside=False)
                elif from_square == Square.H1: # short castling
                    castling = castling._replace(white_kingside=False)
            elif piece.color == BLACK:
                if from_square == Square.A8: # long castling
                    castling = castling._replace(black_queenside=False)
                elif from_square == Square.H8: # short castling
                    castling = castling._replace(black_kingside=False)

        # handle enpassant
        if piece.kind == PAWN:
            if rankdiff == 2: # creating an enpassant opportunity
                enpassant = from_square + 8
            elif rankdiff == -2:
                enpassant = from_square - 8
            elif abs(filediff) == 1 and self.flat_placement[to_square] is None: # capturing enpassant
                if rankdiff > 0: # white eating, a priori
                    eaten = to_square - 8
                else:
                    eaten = to_square + 8
                flat_placement[eaten] = None

        # handle promotion
        if promotion is not None:
            piece = Piece(promotion, piece.color)

        # handle castling
        if (piece.kind == KING) and (abs(filediff) == 2):
            a_side = filediff < 0
            if a_side:
                rook = self.flat_placement[from_square-4]
                flat_placement[from_square-4] = None
                flat_placement[from_square-1] = rook
            else:
                rook = self.flat_placement[from_square+3]
                flat_placement[from_square+3] = None
                flat_placement[from_square+1] = rook

        # handle normal move of the piece
        flat_placement[to_square] = piece

        return Board(tuple(flat_placement), Color(not self.active), castling, enpassant, halfclock, fullclock)

    def algebraic_suffix(self, move: Move, color: Color|None = None, dagger=True):
        """
        returns the suffix of the move in algebraic notation
        only checks for the given color if passed, otherwise figures it out from the move
        """
        if color is None:
            moving_piece = self.flat_placement[move.from_square]
            if moving_piece is None:
                # invalid move
                return ""
            color = Color(not moving_piece.color)
        after = self.make_move(move)
        if after.is_check(color):
            if after.is_stalemate(color):
                return "‡" if dagger else "#"
            return "†" if dagger else "+"
        return ""

    def algebraic_notation_tuple(self, move: Move, long=False, pawn=False, figure=False):
        """
        Returns the algebraic notation tuple of the move, containing
        (1) the moving piece, (2) the starting square, (3) the capture flag, (4) the destination square, (5) the promotion, (6) check/mate/misc suffix,
        (1) will be empty if pawn is False
        (1) will be a unicode figure if figure is True, a cased letter otherwise
        (2) may be empty or only contain one character, if long is False
        (3) will be x or empty
        (5) will be empty if there is no promotion
        (6) will be empty if there is no miscellanous information

        The move is expected to take place *from* the current state of the board, not to arrive *to* it.
        You can apply move.reverse() on the board to get the previous state-ish and get the algebraic notation from it.
        Ish, because if the move was a taking one that information will be lost.
        """
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion
        piece = self.flat_placement[from_square]
        if piece is None:
            # raise ValueError("No piece on square, invalid move")
            piece = Piece(PAWN, self.active) # type: ignore

        if pawn or (piece.kind != PAWN):
            if figure:
                piece_letter = piece.unicode_symbol()
            else:
                piece_letter = piece.symbol()
        else:
            piece_letter = ""

        capture = ""
        if self.flat_placement[to_square] is not None:
            # TODO: en passant
            capture = "x"

        destination = to_square.name.lower()

        if promotion is None:
            promotion = ""
        else:
            promotion = promotion.value.upper()

        suffix = self.algebraic_suffix(move)

        if long or not self.is_legal(move):
            # simple case
            # illegal moves aren't implicit, so they can't be shortened
            origin = from_square.name.lower()
        else:
            left = right = ""
            from_file, from_rank = from_square.indexes()
            movenoprom = move.replace(promotion=None)
            # got to figure out if there was another legal way to get there
            for gen_move in self.generate_moves():
                if gen_move.to_square != to_square:
                    continue

                if gen_move == movenoprom:
                    continue

                if not self.is_legal(gen_move):
                    continue

                gen_from_file, gen_from_rank = gen_move.from_square.indexes()
                if gen_from_file == from_file:
                    print("file", gen_move)
                    left = FILE_NAMES[from_file]
                if gen_from_rank == from_rank:
                    print("rank", gen_move)
                    right = RANK_NAMES[from_rank]

                if left and right:
                    break

            origin = left + right

        return piece_letter, origin, capture, destination, promotion, suffix

    def algebraic_notation(self, move: Move, long=False, pawn=False, figure=False):
        piece = self.flat_placement[move.from_square]
        if (piece is not None) and (piece.kind == KING) and abs(move.from_square.value - move.to_square.value) == 2:
            # castling
            suffix = self.algebraic_suffix(move)
            if move.to_square in (Square.G1, Square.G8):
                return "0-0" + suffix
            elif move.to_square in (Square.C1, Square.C8):
                return "0-0-0" + suffix

        return "".join(self.algebraic_notation_tuple(move, long=long, pawn=pawn, figure=figure))

    def enumerate_raw(self, skip_empty=False):
        for sqn, piece in enumerate(self.flat_placement):
            square = Square(sqn)
            if skip_empty and (piece is None):
                continue
            yield square, *square.indexes(), piece

    def enumerate(self, skip_empty=False):
        # iterate in reverse rank order
        for rankidx in range(7, -1, -1):
            for fileidx in range(8):
                square = Square.square(fileidx, rankidx)
                piece = self.flat_placement[square]
                if skip_empty and (piece is None):
                    continue
                yield square, fileidx, rankidx, piece

Board.empty = Board((None,)*64, None, Castling(False, False, False, False), None) # type: ignore

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
Board.initial = Board.from_fen(START_FEN) # type: ignore
