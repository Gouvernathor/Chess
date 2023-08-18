import renpy # type: ignore
python_object = object
"""renpy
init 1 python in chess.hex:
"""
_constant = True

import collections
import dataclasses
import enum
import functools
import itertools

from store import chess # type: ignore

class Color(enum.IntEnum):
    BLACK = 0
    WHITE = 1
    RED = 0
    GREEN = 1
    BLUE = 2

WHITE_BOARD_COLOR = "#f0d9b5"
GREY_BOARD_COLOR = "#b58863"
BLACK_BOARD_COLOR = "#7a4830"

from store.chess import PieceType # type: ignore

@dataclasses.dataclass(frozen=True)
class HexVector(python_object):
    q: int
    r: int
    @property
    @functools.lru_cache()
    def s(self) -> int:
        return -self.q - self.r

    @property
    def qr(self) -> tuple[int, int]:
        return (self.q, self.r)
    @property
    def qrs(self) -> tuple[int, int, int]:
        return (self.q, self.r, self.s)

    @property
    def vector(self):
        return HexVector(self.q, self.r)

    def isonboard(self):
        return Hex.isonboard(self.q, self.r)

    def __add__(self, other, /):
        return HexVector(self.q + other.q, self.r + other.r)
    def __radd__(self, other, /):
        return self + other
    def __sub__(self, other, /):
        return self + (-other)
    def __rsub__(self, other, /):
        return -self + other
    def __mul__(self, other, /):
        if isinstance(other, int):
            return HexVector(self.q * other, self.r * other)
        return NotImplemented
    __rmul__ = __mul__
    def __pos__(self, /):
        return self
    def __neg__(self, /):
        return -1 * self
    def __abs__(self, /):
        return max(abs(self.q), abs(self.r), abs(self.q+self.r))

class Directions(HexVector, enum.Enum):
    TOP = (0, -1)
    TOPRIGHT = (1, -1)
    BOTTOMRIGHT = (1, 0)
    BOTTOM = (0, 1)
    BOTTOMLEFT = (-1, 1)
    TOPLEFT = (-1, 0)
    def __eq__(self, other, /):
        if not isinstance(other, HexVector):
            return NotImplemented
        return self.qr == other.qr

class Hex(HexVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.isonboard(self.q, self.r):
            raise ValueError(f"Hex out of bounds : {self}")

    @staticmethod
    def isonboard(q, r, s=None):
        if s is not None:
            if q+r+s != 0:
                return False

        if (q < 0) or (r < 0):
            return False

        idx = Hex.index(q, r)
        if idx >= 121 or not Board.STORAGE_MASK[idx]:
            return False

        return True

    @staticmethod
    def index(q, r):
        # we're wrapping at 11
        return q + 11*r

    def is_neighbor(self, other):
        return abs(self - other) == 1

    def get_neighbors(self, dist: int = 1):
        """
        Gets every valid Hex within a distance of dist from self.
        """
        selfq, selfr = self.qr
        for q in range(-dist, dist+1):
            for r in range(max(-dist, -q-dist), min(dist, -q+dist)+1):
                try:
                    rv = Hex(selfq+q, selfr+r)
                except ValueError:
                    pass
                else:
                    yield rv

    def get_at_distance(self, dist: int):
        """
        Gets every valid Hex at a distance of exactly dist from self.
        """
        raise NotImplementedError

    def distance(self, other):
        return abs(self.vector - other.vector)

    def raw_index(self):
        return self.index(self.q, self.r)
    __index__ = raw_index
    @classmethod
    def fromindex(cls, index):
        return cls(index % 11, index // 11)

    @classmethod
    def range(cls):
        """
        Very complex method, conceptually.
        Generates the Hexes in the order they are stored internally.
        """
        for k in itertools.compress(itertools.count(), Board.STORAGE_MASK):
            yield cls.fromindex(k)

    def __str__(self):
        return str(self.qrs)

    # disable these operations between Hexes, but allow them with HexVectors
    def __add__(self, other, /):
        if isinstance(other, Hex):
            return NotImplemented
        rv = super().__add__(other)
        if rv is NotImplemented:
            return NotImplemented
        return type(self)(rv.q, rv.r)
    def __sub__(self, other, /):
        if isinstance(other, Hex):
            return HexVector(self.q - other.q, self.r - other.r)
        return super().__sub__(other)
    __mul__ = (lambda *args: NotImplemented)
    __pos__ = __neg__ = None # type: ignore

from store.chess import Piece # type: ignore

@dataclasses.dataclass(frozen=True)
class Move(python_object):
    from_hex: Hex
    to_hex: Hex
    promotion: PieceType|None = None
    def __str__(self):
        rv = f"{self.from_hex.qrs} -> {self.to_hex.qrs}"
        if self.promotion is not None:
            rv += f" ({self.promotion.value})"
        return rv
    replace = dataclasses.replace

    reverse = None

    @property
    def hex_vector(self):
        return self.to_hex - self.from_hex

@dataclasses.dataclass(frozen=True)
class Board(python_object):
    storage: tuple[Piece|None, ...] # length 121
    # qr representation : using a wrapping 1d storage,
    # starting from the top left, going right,
    # then wrapping under until the bottom right
    # The top-right corner of that 2d grid is the topright corner of the hex board,
    # same for the bottom-left corner
    # The top-left corner of the hex board is the start of the first non-null-starting line,
    # which is also the only line with no null inside
    # (null not meaning None like a simple empty tile,
    # but meaning that that 1d/2d entry has no corresponding tile)
    # The top tile is at the middle of the first 2d line, of which it is also the first non-null element,
    # and same for the bottom tile at the bottom.
    # Going vertically in 2d means going straignt vertically in hex.
    # Going horizontally in 2d means going slightly down and to the right in hex.
    # Going slightly up and to the right in hex means going in slash-diagonal in 2d.
    STORAGE_MASK = (
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    )

    active: Color|None
    enpassant: Hex|None
    halfclock: int = 0
    fullclock: int = 1

    def __post_init__(self):
        storage = self.storage
        try:
            lenstorage = len(storage)
        except TypeError:
            storage = tuple(storage)
            lenstorage = len(storage)

        if lenstorage == 121:
            if isinstance(self.storage, tuple):
                return
            storage = tuple(storage)
        elif lenstorage == 91:
            itstorage = iter(storage)
            storage = tuple(next(itstorage) if flag else None for flag in Board.STORAGE_MASK)
        else:
            raise ValueError(f"Invalid storage length : {lenstorage}. Expected 121 or 91.")

        python_object.__setattr__(self, "storage", storage)

    # @classmethod
    # def one_at_center(cls, piece: Piece|PieceType):
    #     if isinstance(piece, PieceType):
    #         piece = Piece(piece, Color.WHITE)
    #     storage = [None]*121
    #     storage[Hex(5, 5)] = piece
    #     return cls(tuple(storage), None, None)

    @classmethod
    def from_dict_placement(cls, placement: dict[Hex|tuple[int, int], Piece|None], *args, **kwargs):
        storage = [None]*121
        for hex, piece in placement.items():
            if isinstance(hex, tuple):
                hex = Hex.index(*hex)
            storage[hex] = piece
        return cls(tuple(storage), *args, **kwargs)

    def to_dict_placement(self):
        rv = {}
        for hex in Hex.range():
            piece = self.storage[hex]
            if piece is not None:
                rv[hex] = piece
        return rv

    def king_hex(self, color: Color):
        found = Piece(PieceType.KING, color)
        for hex in Hex.range():
            if self.storage[hex] == found:
                return hex
        raise ValueError(f"King not found for {color.name}")

    def pieces_counter(self):
        rv = collections.Counter(itertools.compress(self.storage, Board.STORAGE_MASK))
        del rv[None]
        return rv

    def is_zeroing(self, move: Move):
        if self.storage[move.from_hex].kind == PieceType.PAWN: # type: ignore
            return True
        if self.storage[move.to_hex] is not None:
            return True
        return False

    @staticmethod
    def is_promotion_hex(hex: Hex, color:Color|None = None):
        """
        Only supports black/white boards, for the moment.
        Checks all the promotion tiles, or only those of a given color if given one.
        """
        if color in (Color.BLACK, None):
            if hex.r == 10 or hex.s == -15:
                return True
        if color in (Color.WHITE, None):
            if hex.s == -5 or hex.r == 0:
                return True
        return False

    def generate_moves(self, hex: Hex|None = None):
        if hex is None:
            for hex in Hex.range():
                piece = self.storage[hex]
                if piece is not None and piece.color == self.active:
                    yield from self.generate_moves(hex)
            return

        piece = self.storage[hex]
        if piece is None:
            return
        kind = piece.kind
        color = piece.color
        q, r, s = hex.qrs
        hexvector = hex.vector
        BLACK, WHITE = Color.BLACK, Color.WHITE

        if kind == PieceType.PAWN:
            if self.is_promotion_hex(hex, color):
                # not supposed to happen if promotion is necesarily to a non-pawn kind
                # but just in case, to avoid oob
                return

            enpassant = self.enpassant
            direction = Directions.TOP if color == WHITE else Directions.BOTTOM

            # simple
            if self.storage[hex+direction] is None:
                yield Move(hex, hex+direction)

                # double
                if (((s==-11) if color==WHITE else (r==4)) and (q <= 5) or ((r==6) if color==WHITE else (s==-9)) and (q >= 5))\
                        and self.storage[hex+direction*2] is None:
                    yield Move(hex, hex+direction*2)

            if color == WHITE:
                leftcapture = Directions.TOPLEFT
                rightcapture = Directions.TOPRIGHT
            else:
                leftcapture = Directions.BOTTOMLEFT
                rightcapture = Directions.BOTTOMRIGHT

            # capture
            for capture in (leftcapture, rightcapture):
                if (capture+hexvector).isonboard():
                    destination = hex+capture
                    target = self.storage[destination]
                    if (enpassant == destination) or (target is not None and target.color != color):
                        yield Move(hex, destination)

        DIAGONALS = (
            Directions.TOP + Directions.TOPRIGHT,
            Directions.TOPRIGHT + Directions.BOTTOMRIGHT,
            Directions.BOTTOMRIGHT + Directions.BOTTOM,
            Directions.BOTTOM + Directions.BOTTOMLEFT,
            Directions.BOTTOMLEFT + Directions.TOPLEFT,
            Directions.TOPLEFT + Directions.TOP,
        )

        if kind == PieceType.KING:
            # no castling

            for direction in itertools.chain(Directions, DIAGONALS):
                if (direction+hexvector).isonboard():
                    destination = hex+direction
                    target = self.storage[destination]
                    if target is None or target.color != color:
                        yield Move(hex, destination)

        if kind in (PieceType.ROOK, PieceType.QUEEN):
            for direction in Directions:
                for reach in range(1, 11):
                    if not (reach*direction+hexvector).isonboard():
                        break
                    destination = hex+reach*direction
                    target = self.storage[destination]
                    if target is None:
                        yield Move(hex, destination)
                    elif target.color != color:
                        yield Move(hex, destination)
                        break
                    else:
                        break

        if kind in (PieceType.BISHOP, PieceType.QUEEN):
            for direction in DIAGONALS:
                for reach in range(1, 6):
                    if not (reach*direction+hexvector).isonboard():
                        break
                    destination = hex+reach*direction
                    target = self.storage[destination]
                    if target is None:
                        yield Move(hex, destination)
                    elif target.color != color:
                        yield Move(hex, destination)
                        break
                    else:
                        break

        if kind == PieceType.KNIGHT:
            for direction in (2*Directions.TOP+Directions.TOPRIGHT,
                              2*Directions.TOPRIGHT+Directions.TOP,
                              2*Directions.TOPRIGHT+Directions.BOTTOMRIGHT,
                              2*Directions.BOTTOMRIGHT+Directions.TOPRIGHT,
                              2*Directions.BOTTOMRIGHT+Directions.BOTTOM,
                              2*Directions.BOTTOM+Directions.BOTTOMRIGHT,
                              2*Directions.BOTTOM+Directions.BOTTOMLEFT,
                              2*Directions.BOTTOMLEFT+Directions.BOTTOM,
                              2*Directions.BOTTOMLEFT+Directions.TOPLEFT,
                              2*Directions.TOPLEFT+Directions.BOTTOMLEFT,
                              2*Directions.TOPLEFT+Directions.TOP,
                              2*Directions.TOP+Directions.TOPLEFT,
                              ):
                if (direction+hexvector).isonboard():
                    destination = hex+direction
                    target = self.storage[destination]
                    if target is None or target.color != color:
                        yield Move(hex, destination)

    def make_move(self, move: Move):
        from_hex = move.from_hex
        to_hex = move.to_hex
        promotion = move.promotion

        enpassant = None
        halfclock = self.halfclock + 1
        fullclock = self.fullclock + 1
        if self.active == Color.BLACK:
            fullclock += 1
        storage = list(self.storage)

        if self.is_zeroing(move):
            halfclock = 0

        piece = storage[from_hex]
        if piece is None:
            raise ValueError("No piece on hex, invalid move")
        storage[from_hex] = None

        # no castling

        # handle enpassant
        if piece.kind == PieceType.PAWN:
            move_vector = move.hex_vector
            # creating an enpassant opportunity
            if move_vector == Directions.TOP*2:
                enpassant = from_hex + Directions.TOP
            elif move_vector == Directions.BOTTOM*2:
                enpassant = from_hex + Directions.BOTTOM
            # capturing enpassant
            elif move_vector not in (Directions.TOP, Directions.BOTTOM) and storage[to_hex] is None:
                if move_vector in (Directions.TOPLEFT, Directions.TOPRIGHT):
                    eaten = to_hex + Directions.BOTTOM
                else:
                    eaten = to_hex + Directions.TOP
                storage[eaten] = None

        # handle promotion
        if promotion is not None:
            piece = Piece(promotion, piece.color)

        # no castling

        # handle normal move of the piece
        storage[to_hex] = piece

        return Board(tuple(storage), Color(not self.active), enpassant, halfclock, fullclock)

    def is_legal(self, move: Move, check_check=True, check_turn=True):
        from_hex = move.from_hex
        to_hex = move.to_hex
        promotion = move.promotion

        # check the moving piece
        piece = self.storage[from_hex]
        if piece is None:
            return False

        # check the turn
        if check_turn and piece.color != self.active:
            return False

        # check promotion
        if promotion is not None:
            if piece.kind != PieceType.PAWN:
                return False
            if piece.color == Color.WHITE:
                if to_hex.s != -5 and to_hex.r != 0:
                    return False
            else:
                if to_hex.r != 10 and to_hex.s != -15:
                    return False
            if promotion in (PieceType.KING, PieceType.PAWN):
                return False

        # check the destination
        target = self.storage[to_hex]
        if target is not None:
            if target.color == piece.color:
                return False

        # kind-specific check
        if move.replace(promotion=None) not in self.generate_moves(from_hex):
            return False

        # check self-check
        if check_check:
            if self.make_move(move).is_check(piece.color):
                return False

        return True

    def generate_legal_moves(self, hex: Hex|None = None):
        for move in self.generate_moves(hex):
            if self.is_legal(move):
                yield move

    def taken_hex(self, move: Move):
        """
        Returns a hex this move would take, or None.
        Supports invalid moves where no piece is present to make it.
        Does not check for invalidity when moving and target pieces are of the same color.
        """
        # simple case
        to_hex = move.to_hex
        if self.storage[to_hex] is not None:
            return to_hex

        # enpassant
        from_hex = move.from_hex
        attacker = self.storage[from_hex]
        if attacker is not None and attacker.kind == PieceType.PAWN:
            if to_hex == self.enpassant:
                move_vector = move.hex_vector
                if move_vector in (Directions.TOPLEFT, Directions.TOPRIGHT):
                    return to_hex + Directions.BOTTOM
                elif move_vector in (Directions.BOTTOMLEFT, Directions.BOTTOMRIGHT):
                    return to_hex + Directions.TOP

        return None

    def generate_attackers(self, hex: Hex, color: Color):
        """
        generate moves by other colors which attack the given hex
        moves that would put or leave the opposite king in check are allowed
        enpassant moves are not generated
        """
        for it_hex in Hex.range():
            piece = self.storage[it_hex]
            if (piece is None) or (piece.color == color):
                continue
            for move in self.generate_moves(it_hex):
                if self.taken_hex(move) == hex and self.is_legal(move, check_check=False, check_turn=False):
                    yield move
                    break

    def is_under_attack(self, hex: Hex, color: Color):
        try:
            next(self.generate_attackers(hex, color))
        except StopIteration:
            return False
        else:
            return True

    def generate_checkers(self, color: Color):
        """
        generate moves by other colors which attack the king
        moves that would put or leave the opposite king in check are allowed
        """
        yield from self.generate_attackers(self.king_hex(color), color)

    def is_check(self, color: Color|None = None):
        if color is None:
            return self.is_check(Color.WHITE) or self.is_check(Color.BLACK)

        try:
            next(self.generate_checkers(color))
        except StopIteration:
            return False
        else:
            return True

    def is_stalemate(self, color: Color|None = None):
        if color is None:
            return self.is_stalemate(Color.WHITE) and self.is_stalemate(Color.BLACK)

        for hxn, piece in enumerate(self.storage):
            if self.STORAGE_MASK[hxn]:
                hex = Hex.fromindex(hxn)
                if piece is not None and piece.color == color:
                    for move in self.generate_legal_moves(hex):
                        if not self.make_move(move).is_check(color):
                            return False

        return True

    def is_checkmate(self, color: Color|None = None):
        return self.is_check(color) and self.is_stalemate(color)

    def algebraic_suffix(self, move: Move, color: Color|None = None, dagger=True):
        """
        Should check all colors except the one moving (to support 3+ colors),
        but currently only checks the one other color between black and white.
        In any case a check sign on a move mark when playing 3+ colors doesn't
        mean much if you don't know what color it checks.
        """
        if color is None:
            moving_piece = self.storage[move.from_hex]
            if moving_piece is None:
                # invalid move
                return ""
            color = Color(not moving_piece.color)
        after = self.make_move(move)
        if after.is_check(color):
            if after.is_checkmate(color):
                return "‡" if dagger else "#"
            return "†" if dagger else "+"
        return ""

    def notation(self, move: Move, long=False, pawn=False, figure=False):
        """
        Returns the string notation of the move, containing
        (1) the moving piece, (2) the starting hex, (3) the capture sign, (4) the destination hex, (5) the promotion, (6) the check/mate/misc suffix
        (1) will be empty for pawns if pawn is False
        (1) will be a unicode figure if figure is True, a cased letter otherwise
        (2) may be nothing or only contain one character, if long is False (not currently implemented)
        (3) will be x or nothing
        (5) will be nothing if there is no promotion
        (6) will be nothing if there is no miscellanous information
        """
        from_hex = move.from_hex
        to_hex = move.to_hex
        promotion = move.promotion
        piece = self.storage[from_hex]
        if piece is None:
            # invalid move
            piece = Piece(PieceType.PAWN, self.active)

        if pawn or (piece.kind != PieceType.PAWN):
            if figure:
                piece_letter = piece.unicode_symbol()
            else:
                piece_letter = piece.symbol()
        else:
            piece_letter = ""

        capture = ""
        if self.taken_hex(move) is not None:
            capture = "x"

        destination = to_hex.qrs

        if promotion is None:
            promotion = ""
        else:
            promotion = promotion.value.upper()

        suffix = self.algebraic_suffix(move)

        if True or long or not self.is_legal(move):
            origin = from_hex.qrs

        return f"{piece_letter}{origin}{capture}{destination}{promotion}{suffix}"

Board.empty = Board((None,)*91, None, None)
Board.initial = Board.from_dict_placement(
    {
        (5, 0) : Piece(PieceType.BISHOP, Color.BLACK),
        (5, 1) : Piece(PieceType.BISHOP, Color.BLACK),
        (5, 2) : Piece(PieceType.BISHOP, Color.BLACK),
        (4, 1) : Piece(PieceType.QUEEN, Color.BLACK),
        (6, 0) : Piece(PieceType.KING, Color.BLACK),
        (3, 2) : Piece(PieceType.KNIGHT, Color.BLACK),
        (7, 0) : Piece(PieceType.KNIGHT, Color.BLACK),
        (2, 3) : Piece(PieceType.ROOK, Color.BLACK),
        (8, 0) : Piece(PieceType.ROOK, Color.BLACK),
        (1, 4) : Piece(PieceType.PAWN, Color.BLACK),
        (2, 4) : Piece(PieceType.PAWN, Color.BLACK),
        (3, 4) : Piece(PieceType.PAWN, Color.BLACK),
        (4, 4) : Piece(PieceType.PAWN, Color.BLACK),
        (5, 4) : Piece(PieceType.PAWN, Color.BLACK),
        (6, 3) : Piece(PieceType.PAWN, Color.BLACK),
        (7, 2) : Piece(PieceType.PAWN, Color.BLACK),
        (8, 1) : Piece(PieceType.PAWN, Color.BLACK),
        (9, 0) : Piece(PieceType.PAWN, Color.BLACK),

        (5, 10) : Piece(PieceType.BISHOP, Color.WHITE),
        (5, 9) : Piece(PieceType.BISHOP, Color.WHITE),
        (5, 8) : Piece(PieceType.BISHOP, Color.WHITE),
        (4, 10) : Piece(PieceType.QUEEN, Color.WHITE),
        (6, 9) : Piece(PieceType.KING, Color.WHITE),
        (3, 10) : Piece(PieceType.KNIGHT, Color.WHITE),
        (7, 8) : Piece(PieceType.KNIGHT, Color.WHITE),
        (2, 10) : Piece(PieceType.ROOK, Color.WHITE),
        (8, 7) : Piece(PieceType.ROOK, Color.WHITE),
        (1, 10) : Piece(PieceType.PAWN, Color.WHITE),
        (2, 9) : Piece(PieceType.PAWN, Color.WHITE),
        (3, 8) : Piece(PieceType.PAWN, Color.WHITE),
        (4, 7) : Piece(PieceType.PAWN, Color.WHITE),
        (5, 6) : Piece(PieceType.PAWN, Color.WHITE),
        (6, 6) : Piece(PieceType.PAWN, Color.WHITE),
        (7, 6) : Piece(PieceType.PAWN, Color.WHITE),
        (8, 6) : Piece(PieceType.PAWN, Color.WHITE),
        (9, 6) : Piece(PieceType.PAWN, Color.WHITE),
    },
    Color.WHITE,
    None,
)
