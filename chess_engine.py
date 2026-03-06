#!/usr/bin/env python3
"""A feature-rich UCI chess engine implemented in pure Python.

The engine includes:
- Full legal move generation (castling, en-passant, promotions)
- Negamax alpha-beta search with iterative deepening
- Quiescence search
- Transposition table with Zobrist hashing
- Killer and history move ordering
- UCI protocol support
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse
import math
import random
import sys
import time

# 0x88 board representation constants
EMPTY = 0
WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = -1, -2, -3, -4, -5, -6

WHITE = 1
BLACK = -1

START_FEN = "rn1qkbnr/pppbpppp/8/3p4/8/2NP4/PPPQPPPP/R1B1KBNR w KQkq - 0 1"
# Use standard start unless overridden in commands.
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PIECE_TO_CHAR = {
    WP: "P", WN: "N", WB: "B", WR: "R", WQ: "Q", WK: "K",
    BP: "p", BN: "n", BB: "b", BR: "r", BQ: "q", BK: "k",
    EMPTY: ".",
}

CHAR_TO_PIECE = {v: k for k, v in PIECE_TO_CHAR.items() if k != EMPTY}

PIECE_VALUES = {
    WP: 100, WN: 320, WB: 330, WR: 500, WQ: 900, WK: 0,
    BP: -100, BN: -320, BB: -330, BR: -500, BQ: -900, BK: 0,
}

MATE_SCORE = 30_000
INFINITY = 1_000_000

KNIGHT_OFFSETS = (31, 33, 14, 18, -31, -33, -14, -18)
BISHOP_OFFSETS = (15, 17, -15, -17)
ROOK_OFFSETS = (1, -1, 16, -16)
KING_OFFSETS = (1, -1, 16, -16, 15, 17, -15, -17)

CASTLE_WK = 1
CASTLE_WQ = 2
CASTLE_BK = 4
CASTLE_BQ = 8

PROMOTION_PIECES = (WQ, WR, WB, WN)


PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]
KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
ROOK_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]
QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]
KING_TABLE_MID = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

PIECE_SQUARE = {
    WP: PAWN_TABLE,
    WN: KNIGHT_TABLE,
    WB: BISHOP_TABLE,
    WR: ROOK_TABLE,
    WQ: QUEEN_TABLE,
    WK: KING_TABLE_MID,
    BP: PAWN_TABLE,
    BN: KNIGHT_TABLE,
    BB: BISHOP_TABLE,
    BR: ROOK_TABLE,
    BQ: QUEEN_TABLE,
    BK: KING_TABLE_MID,
}


@dataclass(frozen=True)
class Move:
    from_sq: int
    to_sq: int
    promotion: int = EMPTY
    is_ep: bool = False
    is_castle: bool = False

    def uci(self) -> str:
        s = sq_to_str(self.from_sq) + sq_to_str(self.to_sq)
        if self.promotion:
            s += PIECE_TO_CHAR[abs(self.promotion)].lower()
        return s


@dataclass
class Undo:
    move: Move
    captured: int
    castling: int
    ep_square: int
    halfmove: int
    fullmove: int
    king_w: int
    king_b: int
    zobrist: int


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: int  # 0 exact, -1 alpha, 1 beta
    move: Optional[Move]


class Position:
    def __init__(self, fen: str = START_FEN):
        self.board = [EMPTY] * 128
        self.side = WHITE
        self.castling = 0
        self.ep_square = -1
        self.halfmove = 0
        self.fullmove = 1
        self.king_sq = {WHITE: -1, BLACK: -1}

        # Zobrist
        rnd = random.Random(20250306)
        self.z_piece = {
            p: [rnd.getrandbits(64) for _ in range(128)]
            for p in (WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK)
        }
        self.z_side = rnd.getrandbits(64)
        self.z_castle = [rnd.getrandbits(64) for _ in range(16)]
        self.z_ep = [rnd.getrandbits(64) for _ in range(128)]
        self.zobrist = 0

        self.set_fen(fen)

    def set_fen(self, fen: str) -> None:
        parts = fen.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Invalid FEN: {fen}")

        self.board = [EMPTY] * 128
        placement, side, castling, ep = parts[:4]
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1

        ranks = placement.split("/")
        if len(ranks) != 8:
            raise ValueError("Invalid board in FEN")

        for r, rank in enumerate(ranks):
            file = 0
            for ch in rank:
                if ch.isdigit():
                    file += int(ch)
                else:
                    sq = ((7 - r) << 4) | file
                    piece = CHAR_TO_PIECE[ch]
                    self.board[sq] = piece
                    if piece == WK:
                        self.king_sq[WHITE] = sq
                    elif piece == BK:
                        self.king_sq[BLACK] = sq
                    file += 1

        self.side = WHITE if side == "w" else BLACK
        self.castling = 0
        if "K" in castling:
            self.castling |= CASTLE_WK
        if "Q" in castling:
            self.castling |= CASTLE_WQ
        if "k" in castling:
            self.castling |= CASTLE_BK
        if "q" in castling:
            self.castling |= CASTLE_BQ

        self.ep_square = -1 if ep == "-" else str_to_sq(ep)
        self.zobrist = self.compute_zobrist()

    def compute_zobrist(self) -> int:
        h = 0
        for sq in range(128):
            if sq & 0x88:
                continue
            p = self.board[sq]
            if p:
                h ^= self.z_piece[p][sq]
        h ^= self.z_castle[self.castling]
        if self.ep_square != -1:
            h ^= self.z_ep[self.ep_square]
        if self.side == BLACK:
            h ^= self.z_side
        return h

    def evaluate(self) -> int:
        score = 0
        for sq in range(128):
            if sq & 0x88:
                continue
            p = self.board[sq]
            if not p:
                continue
            score += PIECE_VALUES[p]
            idx64 = ((sq >> 4) * 8) + (sq & 7)
            if p > 0:
                score += PIECE_SQUARE[p][idx64]
            else:
                mirrored = ((7 - (sq >> 4)) * 8) + (sq & 7)
                score -= PIECE_SQUARE[p][mirrored]
        return score if self.side == WHITE else -score

    def is_attacked(self, sq: int, by_side: int) -> bool:
        # Pawns
        pawn_offsets = (-15, -17) if by_side == WHITE else (15, 17)
        pawn = WP if by_side == WHITE else BP
        for off in pawn_offsets:
            t = sq + off
            if not (t & 0x88) and self.board[t] == pawn:
                return True

        # Knights
        knight = WN if by_side == WHITE else BN
        for off in KNIGHT_OFFSETS:
            t = sq + off
            if not (t & 0x88) and self.board[t] == knight:
                return True

        # Bishops/Queens
        bishop = WB if by_side == WHITE else BB
        queen = WQ if by_side == WHITE else BQ
        for off in BISHOP_OFFSETS:
            t = sq + off
            while not (t & 0x88):
                piece = self.board[t]
                if piece:
                    if piece == bishop or piece == queen:
                        return True
                    break
                t += off

        # Rooks/Queens
        rook = WR if by_side == WHITE else BR
        for off in ROOK_OFFSETS:
            t = sq + off
            while not (t & 0x88):
                piece = self.board[t]
                if piece:
                    if piece == rook or piece == queen:
                        return True
                    break
                t += off

        # King
        king = WK if by_side == WHITE else BK
        for off in KING_OFFSETS:
            t = sq + off
            if not (t & 0x88) and self.board[t] == king:
                return True

        return False

    def in_check(self, side: int) -> bool:
        return self.is_attacked(self.king_sq[side], -side)

    def generate_pseudo_legal(self, captures_only: bool = False) -> List[Move]:
        moves: List[Move] = []
        side = self.side

        for sq in range(128):
            if sq & 0x88:
                continue
            piece = self.board[sq]
            if piece * side <= 0:
                continue

            abs_p = abs(piece)
            if abs_p == 1:  # pawn
                forward = 16 if side == WHITE else -16
                start_rank = 1 if side == WHITE else 6
                promo_rank = 6 if side == WHITE else 1
                captures = (15, 17) if side == WHITE else (-15, -17)

                to = sq + forward
                if not captures_only and not (to & 0x88) and self.board[to] == EMPTY:
                    if (sq >> 4) == promo_rank:
                        for prom in PROMOTION_PIECES:
                            moves.append(Move(sq, to, prom if side == WHITE else -prom))
                    else:
                        moves.append(Move(sq, to))
                        if (sq >> 4) == start_rank:
                            to2 = to + forward
                            if self.board[to2] == EMPTY:
                                moves.append(Move(sq, to2))

                for off in captures:
                    to = sq + off
                    if to & 0x88:
                        continue
                    target = self.board[to]
                    if target * side < 0:
                        if (sq >> 4) == promo_rank:
                            for prom in PROMOTION_PIECES:
                                moves.append(Move(sq, to, prom if side == WHITE else -prom))
                        else:
                            moves.append(Move(sq, to))
                    elif to == self.ep_square:
                        moves.append(Move(sq, to, is_ep=True))

            elif abs_p == 2:  # knight
                for off in KNIGHT_OFFSETS:
                    to = sq + off
                    if to & 0x88:
                        continue
                    target = self.board[to]
                    if target == EMPTY:
                        if not captures_only:
                            moves.append(Move(sq, to))
                    elif target * side < 0:
                        moves.append(Move(sq, to))

            elif abs_p in (3, 4, 5):
                offsets = BISHOP_OFFSETS if abs_p == 3 else ROOK_OFFSETS if abs_p == 4 else BISHOP_OFFSETS + ROOK_OFFSETS
                for off in offsets:
                    to = sq + off
                    while not (to & 0x88):
                        target = self.board[to]
                        if target == EMPTY:
                            if not captures_only:
                                moves.append(Move(sq, to))
                        else:
                            if target * side < 0:
                                moves.append(Move(sq, to))
                            break
                        to += off

            else:  # king
                for off in KING_OFFSETS:
                    to = sq + off
                    if to & 0x88:
                        continue
                    target = self.board[to]
                    if target == EMPTY:
                        if not captures_only:
                            moves.append(Move(sq, to))
                    elif target * side < 0:
                        moves.append(Move(sq, to))

                if not captures_only:
                    if side == WHITE and sq == str_to_sq("e1") and not self.in_check(WHITE):
                        if self.castling & CASTLE_WK:
                            if self.board[str_to_sq("f1")] == EMPTY and self.board[str_to_sq("g1")] == EMPTY:
                                if not self.is_attacked(str_to_sq("f1"), BLACK) and not self.is_attacked(str_to_sq("g1"), BLACK):
                                    moves.append(Move(sq, str_to_sq("g1"), is_castle=True))
                        if self.castling & CASTLE_WQ:
                            if self.board[str_to_sq("d1")] == EMPTY and self.board[str_to_sq("c1")] == EMPTY and self.board[str_to_sq("b1")] == EMPTY:
                                if not self.is_attacked(str_to_sq("d1"), BLACK) and not self.is_attacked(str_to_sq("c1"), BLACK):
                                    moves.append(Move(sq, str_to_sq("c1"), is_castle=True))

                    elif side == BLACK and sq == str_to_sq("e8") and not self.in_check(BLACK):
                        if self.castling & CASTLE_BK:
                            if self.board[str_to_sq("f8")] == EMPTY and self.board[str_to_sq("g8")] == EMPTY:
                                if not self.is_attacked(str_to_sq("f8"), WHITE) and not self.is_attacked(str_to_sq("g8"), WHITE):
                                    moves.append(Move(sq, str_to_sq("g8"), is_castle=True))
                        if self.castling & CASTLE_BQ:
                            if self.board[str_to_sq("d8")] == EMPTY and self.board[str_to_sq("c8")] == EMPTY and self.board[str_to_sq("b8")] == EMPTY:
                                if not self.is_attacked(str_to_sq("d8"), WHITE) and not self.is_attacked(str_to_sq("c8"), WHITE):
                                    moves.append(Move(sq, str_to_sq("c8"), is_castle=True))

        return moves

    def generate_legal(self, captures_only: bool = False) -> List[Move]:
        moves = []
        for mv in self.generate_pseudo_legal(captures_only):
            undo = self.make_move(mv)
            if undo is not None:
                moves.append(mv)
                self.unmake_move(undo)
        return moves

    def make_move(self, move: Move) -> Optional[Undo]:
        frm, to = move.from_sq, move.to_sq
        piece = self.board[frm]
        captured = self.board[to]

        undo = Undo(
            move=move,
            captured=captured,
            castling=self.castling,
            ep_square=self.ep_square,
            halfmove=self.halfmove,
            fullmove=self.fullmove,
            king_w=self.king_sq[WHITE],
            king_b=self.king_sq[BLACK],
            zobrist=self.zobrist,
        )

        # Update Zobrist: clear side/castle/ep baseline then restore later by recompute bits quickly
        if self.side == BLACK:
            self.fullmove += 1

        self.ep_square = -1
        self.halfmove += 1

        # Move piece
        self.board[frm] = EMPTY
        self.board[to] = piece

        # En passant capture
        if move.is_ep:
            cap_sq = to - 16 if self.side == WHITE else to + 16
            captured = self.board[cap_sq]
            self.board[cap_sq] = EMPTY
            undo.captured = captured

        # Promotion
        if move.promotion:
            self.board[to] = move.promotion

        # Castling rook move
        if move.is_castle:
            if to == str_to_sq("g1"):
                self.board[str_to_sq("h1")] = EMPTY
                self.board[str_to_sq("f1")] = WR
            elif to == str_to_sq("c1"):
                self.board[str_to_sq("a1")] = EMPTY
                self.board[str_to_sq("d1")] = WR
            elif to == str_to_sq("g8"):
                self.board[str_to_sq("h8")] = EMPTY
                self.board[str_to_sq("f8")] = BR
            elif to == str_to_sq("c8"):
                self.board[str_to_sq("a8")] = EMPTY
                self.board[str_to_sq("d8")] = BR

        # King positions
        if piece == WK:
            self.king_sq[WHITE] = to
            self.castling &= ~(CASTLE_WK | CASTLE_WQ)
        elif piece == BK:
            self.king_sq[BLACK] = to
            self.castling &= ~(CASTLE_BK | CASTLE_BQ)

        # Rook moves/captures affecting castling rights
        if frm == str_to_sq("h1") or to == str_to_sq("h1"):
            self.castling &= ~CASTLE_WK
        if frm == str_to_sq("a1") or to == str_to_sq("a1"):
            self.castling &= ~CASTLE_WQ
        if frm == str_to_sq("h8") or to == str_to_sq("h8"):
            self.castling &= ~CASTLE_BK
        if frm == str_to_sq("a8") or to == str_to_sq("a8"):
            self.castling &= ~CASTLE_BQ

        # Pawn specifics
        if abs(piece) == 1:
            self.halfmove = 0
            if abs(to - frm) == 32:
                self.ep_square = (frm + to) // 2
        if undo.captured != EMPTY:
            self.halfmove = 0

        self.side = -self.side

        # Reject illegal king exposure
        if self.in_check(-self.side):
            self.unmake_move(undo)
            return None

        self.zobrist = self.compute_zobrist()
        return undo

    def unmake_move(self, undo: Undo) -> None:
        move = undo.move
        frm, to = move.from_sq, move.to_sq

        self.side = -self.side
        self.castling = undo.castling
        self.ep_square = undo.ep_square
        self.halfmove = undo.halfmove
        self.fullmove = undo.fullmove
        self.king_sq[WHITE] = undo.king_w
        self.king_sq[BLACK] = undo.king_b

        piece = self.board[to]
        if move.promotion:
            piece = WP if self.side == WHITE else BP

        self.board[frm] = piece
        self.board[to] = undo.captured

        if move.is_ep:
            self.board[to] = EMPTY
            cap_sq = to - 16 if self.side == WHITE else to + 16
            self.board[cap_sq] = BP if self.side == WHITE else WP

        if move.is_castle:
            if to == str_to_sq("g1"):
                self.board[str_to_sq("h1")] = WR
                self.board[str_to_sq("f1")] = EMPTY
            elif to == str_to_sq("c1"):
                self.board[str_to_sq("a1")] = WR
                self.board[str_to_sq("d1")] = EMPTY
            elif to == str_to_sq("g8"):
                self.board[str_to_sq("h8")] = BR
                self.board[str_to_sq("f8")] = EMPTY
            elif to == str_to_sq("c8"):
                self.board[str_to_sq("a8")] = BR
                self.board[str_to_sq("d8")] = EMPTY

        self.zobrist = undo.zobrist


class Engine:
    def __init__(self):
        self.tt: Dict[int, TTEntry] = {}
        self.nodes = 0
        self.start_time = 0.0
        self.stop_time = 0.0
        self.stop = False
        self.killers: List[List[Optional[Move]]] = [[None, None] for _ in range(128)]
        self.history: Dict[Tuple[int, int], int] = {}

    def time_up(self) -> bool:
        return self.stop or (self.stop_time > 0 and time.time() >= self.stop_time)

    def score_move(self, pos: Position, mv: Move, tt_move: Optional[Move], ply: int) -> int:
        if tt_move and mv == tt_move:
            return 1_000_000

        target = pos.board[mv.to_sq]
        piece = pos.board[mv.from_sq]
        if mv.is_ep:
            target = BP if pos.side == WHITE else WP

        if target:
            return 100_000 + 10 * abs(PIECE_VALUES[target]) - abs(PIECE_VALUES[piece])

        k1, k2 = self.killers[ply]
        if mv == k1:
            return 90_000
        if mv == k2:
            return 80_000

        return self.history.get((mv.from_sq, mv.to_sq), 0)

    def quiescence(self, pos: Position, alpha: int, beta: int, ply: int) -> int:
        if self.time_up():
            return pos.evaluate()

        self.nodes += 1
        stand_pat = pos.evaluate()
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        moves = pos.generate_legal(captures_only=True)
        moves.sort(key=lambda m: self.score_move(pos, m, None, ply), reverse=True)

        for mv in moves:
            undo = pos.make_move(mv)
            if undo is None:
                continue
            score = -self.quiescence(pos, -beta, -alpha, ply + 1)
            pos.unmake_move(undo)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def search(self, pos: Position, depth: int, alpha: int, beta: int, ply: int = 0) -> int:
        if self.time_up():
            self.stop = True
            return 0

        self.nodes += 1
        in_check = pos.in_check(pos.side)
        if depth <= 0:
            return self.quiescence(pos, alpha, beta, ply)

        if ply > 0:
            alpha = max(alpha, -MATE_SCORE + ply)
            beta = min(beta, MATE_SCORE - ply)
            if alpha >= beta:
                return alpha

        tt_move = None
        key = pos.zobrist
        tt_entry = self.tt.get(key)
        if tt_entry and tt_entry.depth >= depth:
            tt_move = tt_entry.move
            if tt_entry.flag == 0:
                return tt_entry.score
            if tt_entry.flag == -1 and tt_entry.score <= alpha:
                return tt_entry.score
            if tt_entry.flag == 1 and tt_entry.score >= beta:
                return tt_entry.score

        moves = pos.generate_legal()
        if not moves:
            if in_check:
                return -MATE_SCORE + ply
            return 0

        moves.sort(key=lambda m: self.score_move(pos, m, tt_move, ply), reverse=True)

        best_move = None
        best_score = -INFINITY
        orig_alpha = alpha

        for idx, mv in enumerate(moves):
            undo = pos.make_move(mv)
            if undo is None:
                continue

            if idx == 0:
                score = -self.search(pos, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Principal Variation Search
                score = -self.search(pos, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self.search(pos, depth - 1, -beta, -alpha, ply + 1)

            pos.unmake_move(undo)

            if self.stop:
                return 0

            if score > best_score:
                best_score = score
                best_move = mv

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if pos.board[mv.to_sq] == EMPTY:
                    self.killers[ply][1] = self.killers[ply][0]
                    self.killers[ply][0] = mv
                    key_hist = (mv.from_sq, mv.to_sq)
                    self.history[key_hist] = self.history.get(key_hist, 0) + depth * depth
                break

        flag = 0
        if best_score <= orig_alpha:
            flag = -1
        elif best_score >= beta:
            flag = 1

        self.tt[key] = TTEntry(depth=depth, score=best_score, flag=flag, move=best_move)
        return best_score

    def iterative_deepening(self, pos: Position, max_depth: int = 8, max_time: float = 5.0) -> Tuple[Optional[Move], int, int]:
        self.nodes = 0
        self.stop = False
        self.start_time = time.time()
        self.stop_time = self.start_time + max_time if max_time > 0 else 0

        best_move = None
        best_score = 0

        for depth in range(1, max_depth + 1):
            score = self.search(pos, depth, -INFINITY, INFINITY)
            if self.stop:
                break
            best_score = score
            tt_entry = self.tt.get(pos.zobrist)
            if tt_entry and tt_entry.move:
                best_move = tt_entry.move

            elapsed_ms = int((time.time() - self.start_time) * 1000)
            nps = int(self.nodes / max(0.001, time.time() - self.start_time))
            pv = best_move.uci() if best_move else ""
            print(f"info depth {depth} score cp {best_score} nodes {self.nodes} nps {nps} time {elapsed_ms} pv {pv}")
            sys.stdout.flush()

        return best_move, best_score, self.nodes


def sq_to_str(sq: int) -> str:
    return chr((sq & 7) + ord("a")) + str((sq >> 4) + 1)


def str_to_sq(s: str) -> int:
    file = ord(s[0]) - ord("a")
    rank = int(s[1]) - 1
    return (rank << 4) | file


def parse_uci_move(pos: Position, text: str) -> Optional[Move]:
    if len(text) < 4:
        return None
    frm = str_to_sq(text[:2])
    to = str_to_sq(text[2:4])
    promo = EMPTY
    if len(text) > 4:
        mp = {"q": WQ, "r": WR, "b": WB, "n": WN}
        base = mp.get(text[4].lower())
        if base is None:
            return None
        promo = base if pos.side == WHITE else -base
    for mv in pos.generate_legal():
        if mv.from_sq == frm and mv.to_sq == to:
            if (not mv.promotion and promo == EMPTY) or (mv.promotion == promo):
                return mv
    return None


def perft(pos: Position, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for mv in pos.generate_legal():
        undo = pos.make_move(mv)
        if undo is None:
            continue
        nodes += perft(pos, depth - 1)
        pos.unmake_move(undo)
    return nodes


def run_uci() -> None:
    pos = Position(START_FEN)
    engine = Engine()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        if line == "uci":
            print("id name GPT-Python-Engine")
            print("id author OpenAI")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line == "ucinewgame":
            pos = Position(START_FEN)
            engine = Engine()
        elif line.startswith("position"):
            parts = line.split()
            idx = 1
            if parts[idx] == "startpos":
                pos = Position(START_FEN)
                idx += 1
            elif parts[idx] == "fen":
                fen_tokens = []
                idx += 1
                while idx < len(parts) and parts[idx] != "moves":
                    fen_tokens.append(parts[idx])
                    idx += 1
                pos = Position(" ".join(fen_tokens))
            if idx < len(parts) and parts[idx] == "moves":
                idx += 1
                while idx < len(parts):
                    mv = parse_uci_move(pos, parts[idx])
                    if mv:
                        undo = pos.make_move(mv)
                        if undo is None:
                            break
                    idx += 1
        elif line.startswith("go"):
            parts = line.split()
            max_depth = 6
            move_time = 3.0
            if "depth" in parts:
                max_depth = int(parts[parts.index("depth") + 1])
                move_time = 0
            elif "movetime" in parts:
                move_time = int(parts[parts.index("movetime") + 1]) / 1000.0
            elif "wtime" in parts and "btime" in parts:
                wtime = int(parts[parts.index("wtime") + 1])
                btime = int(parts[parts.index("btime") + 1])
                rem = wtime if pos.side == WHITE else btime
                move_time = max(0.05, rem / 30_000)

            best_move, _, _ = engine.iterative_deepening(pos, max_depth=max_depth, max_time=move_time)
            if best_move is None:
                legal = pos.generate_legal()
                if legal:
                    best_move = legal[0]
            print(f"bestmove {best_move.uci() if best_move else '0000'}")
        elif line == "stop":
            engine.stop = True
        elif line == "quit":
            break

        sys.stdout.flush()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Python chess engine")
    parser.add_argument("--uci", action="store_true", help="Run UCI loop")
    parser.add_argument("--fen", type=str, default=START_FEN)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time", type=float, default=2.0)
    parser.add_argument("--perft", type=int, default=0)
    args = parser.parse_args()

    if args.uci:
        run_uci()
        return

    pos = Position(args.fen)
    if args.perft > 0:
        print(perft(pos, args.perft))
        return

    engine = Engine()
    move, score, nodes = engine.iterative_deepening(pos, max_depth=args.depth, max_time=args.time)
    print(f"best move: {move.uci() if move else 'none'} score: {score} nodes: {nodes}")


if __name__ == "__main__":
    cli()
