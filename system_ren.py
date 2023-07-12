import renpy # type: ignore
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
        rv = super().__add__(other)
        if rv is NotImplemented:
            return NotImplemented
        return type(self)(rv)
    def __sub__(self, other):
        rv = super().__sub__(other)
        if rv is NotImplemented:
            return NotImplemented
        return type(self)(rv)

UNICODE_PIECE_SYMBOLS = {
    "R": "♖", "r": "♜",
    "N": "♘", "n": "♞",
    "B": "♗", "b": "♝",
    "Q": "♕", "q": "♛",
    "K": "♔", "k": "♚",
    "P": "♙", "p": "♟",
}
@dataclasses.dataclass(frozen=True, order=True)
class Piece:
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
class Move:
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
class Board:
    """
    contains the same information as a FEN
    """

    flat_placement: tuple[Piece|None] # type: tuple[((Piece|None),)*64]
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

    def __getitem__(self, square: tuple[int, int]):
        """
        takes either a square number (0-63) or a tuple (fileidx 0-7, rankidx 0-7)
        """
        return self.flat_placement[Square.square(*square)]

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

    def generate_attackers(self, square: Square, color: Color):
        """
        generates moves by the opposite color which attack the given square
        moves that would put or leave the king of the opposite color in check are allowed (per the rules)
        """
        for it_square in Square.range():
            piece = self.flat_placement[it_square]
            if (piece is None) or (piece.color == color):
                continue
            for move in self.generate_moves(it_square):
                if move.to_square == square and self.is_legal(move, check_check=False, check_turn=False):
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
                for move in self.generate_moves(square):
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

        # check the destination
        dest_piece = self.flat_placement[to_square]
        if dest_piece is not None:
            if dest_piece.color == piece.color:
                return False

        # castling
        if (piece.kind == KING) and not self.is_check(piece.color):
            filediff, rankdiff = move.diff_tuple()
            if abs(filediff)+abs(rankdiff) != 1:
                # check castling
                if move not in self.generate_castling_moves(color=piece.color):
                    return False

        # kind-dependant check
        if move.replace(promotion=None) not in self.generate_moves(from_square, castling=False):
            return False

        if check_check:
            # check if the move puts the king in check
            if self.make_move(move).is_check(piece.color):
                return False

        return True

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
