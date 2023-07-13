import renpy # type: ignore
"""renpy
init 1 python in chess.player:
"""
import collections

from store import chess # type: ignore

class AbstractPlayer:
    def get_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError

class RandomPlayer(AbstractPlayer):
    """
    Should be monotonic with rollback, relies on renpy.random.
    You can change the random object in __init__, for not much added value
    other than independance with other players.
    """
    random = renpy.random

    def get_move(self, board: chess.Board) -> chess.Move:
        pool = list(board.generate_legal_moves())
        return self.random.choice(pool)

class StraightRandomPlayer(RandomPlayer):
    """
    The same instance always generate the same move for a given board,
    no matter the order or repetitions of the get_move call : it's pure.
    """
    def __init__(self, key=None):
        self.key = key or renpy.random.random()

    def get_move(self, board: chess.Board) -> chess.Move:
        randomobj = renpy.random.Random(self.key)
        pool = list(board.generate_legal_moves())
        return randomobj.choice(pool)

class PonderatedRandom1(AbstractPlayer):
    """
    From the list of legal moves, advantages those that capture a piece, mate, check, or promote.
    If I want to spend the time, recursively explores successive moves by the opponent and so on.
    """
    bonus = dict(
        capture={
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        },
        promote=10,
        check=5,
        # mate=float('inf'), # override actually
        par=-15,
    )

    def get_move(self, board: chess.Board) -> chess.Move:
        pool = collections.Counter(board.generate_legal_moves())
        for move in pool:
            eatenpiece = board.flat_placement[move.to_square]
            if eatenpiece is not None:
                pool[move] += self.bonus['capture'][eatenpiece.kind]

            if move.promotion is not None:
                pool[move] += self.bonus['promote']

            after = board.make_move(move)
            if after.is_checkmate(chess.Color(not board.active)):
                # let's simply win
                return move
                # pool[move] += self.bonus['mate']
            elif after.is_check(chess.Color(not board.active)):
                pool[move] += self.bonus['check']
            elif after.is_stalemate(chess.Color(not board.active)):
                pool[move] += self.bonus['par']

        return pool.most_common(1)[0][0]
