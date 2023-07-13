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
import math

from store import chess # type: ignore

class Color(enum.IntEnum):
    WHITE = 0
    BLACK = 1
    RED = 0
    GREEN = 1
    BLUE = 2

WHITE_BOARD_COLOR = "#f0d9b5"
GREY_BOARD_COLOR = "#b58863"
BLACK_BOARD_COLOR = "#7a4830"

from store.chess import PieceType # type: ignore

@dataclasses.dataclass(frozen=True)
class HexVector(python_object):
    q: int = 0
    r: int = 0
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

    def __add__(self, other, /):
        return HexVector(self.q + other.q, self.r + other.r)
    def __radd__(self, other, /):
        return self + other
    def __sub__(self, other, /):
        return self + (-other)
    def __rsub__(self, other, /):
        return -self + other
    def __mul__(self, other, /):
        return HexVector(self.q * other, self.r * other)
    __rmul__ = __mul__
    def __truediv__(self, other, /):
        return self * (1 / other)
    def __pos__(self, /):
        return self
    def __neg__(self, /):
        return -1 * self
    def __abs__(self, /):
        return max(abs(self.q), abs(self.r), abs(self.q+self.r))

class Directions(HexVector, enum.Enum):
    TOP = HexVector(0, -1) # -1 color
    TOP_RIGHT = HexVector(1, -1) # +1 color
    BOTTOM_RIGHT = HexVector(1, 0) # -1 color
    BOTTOM = HexVector(0, 1) # +1 color
    BOTTOM_LEFT = HexVector(-1, 1) # -1 color
    TOP_LEFT = HexVector(-1, 0) # +1 color
    # (-q+r)%3

class Hex(HexVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        idx = self.raw_index()
        if not Board.storage_mask[idx]:
            raise ValueError(f"Hex out of bounds : {self}")

    def is_neighbor(self, other):
        return abs(self - other) == 1

    def get_neighbors(self, dist: int = 1):
        for q in range(-dist, dist+1):
            for r in range(max(-dist, -q-dist), min(dist, -q+dist)+1):
                yield Hex(q, r)

    def raw_index(self):
        # we're wrapping at 11
        return self.q + 11*self.r
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
        for k in itertools.compress(itertools.count(), Board.storage_mask):
            yield cls.fromindex(k)

    # disable these operations between Hexes, but allow them with HexVectors
    def __add__(self, other, /):
        if isinstance(other, Hex):
            return NotImplemented
        rv = super().__add__(other)
        if rv is NotImplemented:
            return NotImplemented
        return type(self)(rv.q, rv.r)
    __mul__ = __truediv__ = __pos__ = __neg__ = (lambda *args: NotImplemented)

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

Castling = NotImplementedError # TODO

@dataclasses.dataclass(frozen=True)
class Board(python_object):
    storage: tuple[Piece|None]
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
    storage_mask = (
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
    castling: Castling
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
            storage = tuple(next(itstorage) if flag else None for flag in Board.storage_mask)
        else:
            raise ValueError(f"Invalid storage length : {lenstorage}. Expected 121 or 91.")

        python_object.__setattr__(self, "storage", storage)

    def king_hex(self, color: Color):
        found = Piece(PieceType.KING, color)
        for hex in Hex.range():
            if self.storage[hex] == found:
                return hex
        raise ValueError(f"King not found for {color.name}")

    def pieces_counter(self):
        rv = collections.Counter(itertools.compress(self.storage, Board.storage_mask))
        del rv[None]
        return rv

Board.empty = Board((None,)*91, None, None, None)
