from json.encoder import INFINITY
from board import *
import chess
import random
import time
from game import *
from move import *
import chess.engine
from multiprocessing import Pool
from typing import Dict, List, Any
import math

class Engine:

    def __init__(self, board, game):
        self.my_turn = False
        self.color = 'black'
        self.good_moves = []
        self.bad_moves = []
        self.board = board
        self.game = game
        self.turns = 0
        self.castled = False
        self.zobroist = []
        self.current_depth = 0
        self.global_best_move = None
        self.timeout = 10000
        self.timeout_bool = False
        self.start_time = 0
        for _ in range(64):
            row = []
            for _ in range(12):
                row.append(random.randrange(2**64))
            self.zobroist.append(row)
        self.table = {}

    def static_eval(self, board_instance):
        eval = 0
        pos = 0
        pos_val = 0
        for i in range(64):
            if board_instance.piece_at(i):
                piece = board_instance.piece_at(i)
                if piece.color == True:
                    numVal = self.get_piece_val(i, piece.symbol())
                    eval += numVal
                else:
                    numVal = -self.get_piece_val(i, piece.symbol())
                    eval += numVal
        
        return eval

    def move_value(self, board: chess.Board, move: chess.Move) -> float:
        """
        How good is a move?
        A promotion is great.
        A weaker piece taking a stronger piece is good.
        A stronger piece taking a weaker piece is bad.
        Also consider the position change via piece-square table.
        """
        if move.promotion is not None:
            return -float("inf") if board.turn == chess.BLACK else float("inf")

        _piece = board.piece_at(move.from_square)
        if _piece:
            _from_value = self.evaluate_piece(_piece, move.from_square)
            _to_value = self.evaluate_piece(_piece, move.to_square)
            position_change = _to_value - _from_value
        else:
            raise Exception(f"A piece was expected at {move.from_square}")

        capture_value = 0.0
        if board.is_capture(move):
            capture_value = self.valuate_capture(board, move)

        current_move_value = capture_value + position_change
        if board.turn == chess.BLACK:
            current_move_value = -current_move_value

        return current_move_value

    def evaluate_piece(self, piece: chess.Piece, square: chess.Square) -> int:
        piece_type = piece.piece_type
        mapping = []
        if piece_type == chess.PAWN:
            mapping = pawnEvalWhite if piece.color == chess.WHITE else pawnEvalBlack
        if piece_type == chess.KNIGHT:
            mapping = knightEval
        if piece_type == chess.BISHOP:
            mapping = bishopEvalWhite if piece.color == chess.WHITE else bishopEvalBlack
        if piece_type == chess.ROOK:
            mapping = rookEvalWhite if piece.color == chess.WHITE else rookEvalBlack
        if piece_type == chess.QUEEN:
            mapping = queenEval
        if piece_type == chess.KING:
            # use end game piece-square tables if neither side has a queen
            mapping = kingEvalWhite if piece.color == chess.WHITE else kingEvalBlack

        return mapping[square]

    def evaluate_board(self, board: chess.Board) -> float:
        """
        Evaluates the full board and determines which player is in a most favorable position.
        The sign indicates the side:
            (+) for white
            (-) for black
        The magnitude, how big of an advantage that player has
        """
        total = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue

            value = piece_value[piece.piece_type] + self.evaluate_piece(piece, square)
            total += value if piece.color == chess.WHITE else -value

        return total
    def record(self, hash, value, decision, depth, flag):
        if hash not in self.table:
            self.table[hash] = {}

        self.table[hash]['value'] = value
        self.table[hash]['decision'] = decision
        self.table[hash]['depth'] = depth
        self.table[hash]['flag'] = flag

    def zobroist_hash(self, board):
        pieces = board.piece_map()
        hash = 0
        for square in chess.SQUARES:
            if square in pieces:
                piece = pieces[square]
                piece_type = piece.piece_type + piece.color * 6 - 1
                hash ^= self.zobroist[square][piece_type]
        return hash

    def evaluate_capture(self, board: chess.Board, move: chess.Move) -> float:
        """
        Given a capturing move, weight the trade being made.
        """
        if board.is_en_passant(move):
            return piece_value[chess.PAWN]
        _to = board.piece_at(move.to_square)
        _from = board.piece_at(move.from_square)
        if _to is None or _from is None:
            raise Exception(
                f"Pieces were expected at _both_ {move.to_square} and {move.from_square}"
            )
        return piece_value[_to.piece_type] - piece_value[_from.piece_type]

    def move(self):
        move  = self.select_move()
        print(move)
        ascii_move_initial = chess.square_name(move.from_square)
        ascii_move_final = chess.square_name(move.to_square)
        initial = Square(INV_SQUARE_NAMES.get(ascii_move_initial)[0], INV_SQUARE_NAMES.get(ascii_move_initial)[1])
        final = Square(INV_SQUARE_NAMES.get(ascii_move_final)[0], INV_SQUARE_NAMES.get(ascii_move_final)[1])
        piece = self.board.squares[initial.row][initial.col].piece
        move = Move(initial, final, piece)
        self.board.move(piece, move)
        start_pos = SQUARE_NAMES[move.initial.row, move.initial.col]
        end_pos = SQUARE_NAMES[move.final.row, move.final.col]
        cboard_move = chess.Move.from_uci(start_pos + end_pos)
        self.game.cboard.push(cboard_move)
        self.board.clear_all_moves()
        self.board.calc_moves(self.game.cboard)
        self.board.next_player = 'white'
        self.turns += 2
        print(counter)

    def select_move(self):
        global counter
        counter = 0
        return self.find_best_move(self.game.cboard)

    def iterative(self, tboard):
        global nextMove
        nextMove = None
        self.timeout_bool = False
        self.global_best_move = None
        self.current_depth = 0
        self.start_time = time.time() * 1000
        for i in range(100):
            if i > 0:
                self.global_best_move = nextMove
                print("Completed search at depth: " + str(self.current_depth))
            print(self.global_best_move)
            self.current_depth = self.current_depth + 1
            self.negamax_ab(tboard, self.current_depth, -1000, 1000, -1)
            if self.timeout_bool:
                return self.global_best_move


    def find_best_move(self, tboard):
        global nextMove
        nextMove = None
        pool = Pool
        #self.negascout(tboard, DEPTH, -1000, 1000, -1)
        #self.negamax_ab(tboard, DEPTH, -1000, 1000, -1)
        #self.negamax(tboard, validMoves, DEPTH, -1)
        return self.iterative(tboard)

    def get_ordered_moves(self, board: chess.Board) -> List[chess.Move]:
        """
        Get legal moves.
        Attempt to sort moves by best to worst.
        Use piece values (and positional gains/losses) to weight captures.
        """

        def orderer(move):
            return self.move_value(board, move)

        in_order = sorted(
            board.legal_moves, key=orderer, reverse=(board.turn == chess.WHITE)
        )
        return list(in_order)

    def move_value(self, board: chess.Board, move: chess.Move) -> float:
        """
        How good is a move?
        A promotion is great.
        A weaker piece taking a stronger piece is good.
        A stronger piece taking a weaker piece is bad.
        Also consider the position change via piece-square table.
        """
        if move.promotion is not None:
            return -float("inf") if board.turn == chess.BLACK else float("inf")

        _piece = board.piece_at(move.from_square)
        if _piece:
            _from_value = self.evaluate_piece(_piece, move.from_square)
            _to_value = self.evaluate_piece(_piece, move.to_square)
            position_change = _to_value - _from_value
        else:
            raise Exception(f"A piece was expected at {move.from_square}")

        capture_value = 0.0
        if board.is_capture(move):
            capture_value = self.evaluate_capture(board, move)

        current_move_value = capture_value + position_change
        if board.turn == chess.BLACK:
            current_move_value = -current_move_value

        return current_move_value

    def qSearch(self, board, alpha, beta, turnMultiplier, startingDepth, depth=0, maxDepth=4):
        global counter
        counter += 1
        if board.is_checkmate():
            return turnMultiplier * (1 - (0.01*(DEPTH - depth))) * -99999 if board.turn else turnMultiplier * (1 - (0.01*(DEPTH - depth))) * 99999
        
        value = turnMultiplier * self.evaluate_board(board)
        if value >= beta:
            return beta
        if alpha < value:
            alpha = value
        if depth < maxDepth:
            captureMoves = (move for move in board.generate_legal_moves() if (board.is_capture(move) or board.is_check()))
            for move in captureMoves:
                board.push(move)
                score = -1 * self.qSearch(board, -beta, -alpha, -turnMultiplier, depth + 1, maxDepth)
                board.pop()
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha

    '''def negascout(self, tboard, depth, alpha, beta, turnMultiplier):
        global nextMove, counter
        alphaOrig = alpha
        counter += 1
        if depth == 0:
            # Quiescence Search - Enable for higher skill engine but longer move times
            #if tboard.is_capture(tboard.peek()) or tboard.is_check():
                #return self.qSearch(tboard, alpha, beta, turnMultiplier, DEPTH)
            return turnMultiplier * self.evaluate_board(tboard)
        if tboard.is_fivefold_repetition() or tboard.is_stalemate() or tboard.is_seventyfive_moves():
            return 0
        if tboard.is_checkmate():
            return turnMultiplier * (1 - (0.01*(DEPTH - depth))) * -99999 if tboard.turn else turnMultiplier * (1 - (0.01*(DEPTH - depth))) * 99999
        maxScore = -100000
        hash = self.zobroist_hash(tboard)
        if hash in self.table:
            if self.table[hash]['depth'] >= depth:
                value = self.table[hash]['value']
                if self.table[hash]['flag'] == 'lower':
                    alpha = max(alpha, value)
                elif self.table[hash]['flag'] == 'upper':
                    beta = max(beta, value)
                elif self.table[hash]['flag'] == 'exact':
                    return value
                if alpha >= beta:
                    return value
        moves = self.get_ordered_moves(tboard)
        n = beta
        for legal_move in moves:
            tboard.push(legal_move)
            score = -self.negascout(tboard, depth - 1, -n, -alpha, -turnMultiplier)
            if score > maxScore:
                if n == beta or depth <= 2:
                    maxScore = score
                else:
                    score = -self.negascout(tboard, depth - 1, -beta, -score, -turnMultiplier)
                if depth == DEPTH:
                    nextMove = legal_move
            tboard.pop()
            if maxScore > alpha:
                alpha = maxScore
            if alpha >= beta:
                return alpha
            n = alpha + 1
        flag = ''
        if maxScore <= alphaOrig:
            flag = 'upper'
        elif maxScore >= beta:
            flag = 'lower'
        else:
            flag = 'exact'
        self.record(hash, maxScore, legal_move, depth, flag)
        return maxScore'''

    def negamax_ab(self, tboard, depth, alpha, beta, turnMultiplier):
        global nextMove, counter
        if (time.time() * 1000) - self.start_time > self.timeout:
            self.timeout_bool = True
            return turnMultiplier * self.evaluate_board(tboard)
        alphaOrig = alpha
        counter += 1
        if depth == 0:
            # Quiescence Search - Enable for higher skilled engine but slower run time
            #if tboard.is_capture(tboard.peek()) or tboard.is_check():
            #    return self.qSearch(tboard, alpha, beta, turnMultiplier, DEPTH)
            return turnMultiplier * self.evaluate_board(tboard)
        if tboard.is_fivefold_repetition() or tboard.is_stalemate() or tboard.is_seventyfive_moves():
            return 0
        if tboard.is_checkmate():
            return turnMultiplier * (1 - (0.01*(DEPTH - depth))) * -99999 if tboard.turn else turnMultiplier * (1 - (0.01*(DEPTH - depth))) * 99999
        maxScore = -100000
        hash = self.zobroist_hash(tboard)
        if hash in self.table:
            if self.table[hash]['depth'] >= depth:
                value = self.table[hash]['value']
                if self.table[hash]['flag'] == 'lower':
                    alpha = max(alpha, value)
                elif self.table[hash]['flag'] == 'upper':
                    beta = max(beta, value)
                elif self.table[hash]['flag'] == 'exact':
                    return value
                if alpha >= beta:
                    return value
        moves = self.get_ordered_moves(tboard)

        for legal_move in moves:
            tboard.push(legal_move)
            score = -self.negamax_ab(tboard, depth - 1, -beta, -alpha, -turnMultiplier)
            if score > maxScore:
                maxScore = score
                if depth == self.current_depth:
                    nextMove = legal_move
            tboard.pop()
            if maxScore > alpha:
                alpha = maxScore
            if alpha >= beta:
                break
        flag = ''
        if maxScore <= alphaOrig:
            flag = 'upper'
        elif maxScore >= beta:
            flag = 'lower'
        else:
            flag = 'exact'
        self.record(hash, maxScore, legal_move, depth, flag)
        return maxScore
