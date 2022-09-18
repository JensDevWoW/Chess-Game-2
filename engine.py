from json.encoder import INFINITY
from board import *
import random
import time
from game import *
from move import *


class Engine:

    def __init__(self, board, game):
        self.my_turn = False
        self.color = 'black'
        self.good_moves = []
        self.bad_moves = []
        self.board = board
        self.game = game
        self.white_pieces = []
        self.black_pieces = []
        self.sort_pieces()

    def move(self):
        move = self.select_move()
        piece = self.board.squares[move.initial.row][move.initial.col].piece
        self.board.move(piece, move)
        self.board.clear_all_moves()
        self.board.next_player = 'white'
        print(counter)

    def sort_pieces(self):
        self.white_pieces = []
        self.black_pieces = []
        for row in range(ROWS):
            for col in range(COLS):
                if self.board.squares[row][col].has_piece():
                    piece = self.board.squares[row][col].piece
                    if piece.color == 'white':
                        self.white_pieces.append(piece)
                    else:
                        self.black_pieces.append(piece)

    def select_move(self):
        global counter
        tboard = copy.deepcopy(self.board)
        validMoves = self.board.getValidMoves()
        random.shuffle(validMoves)
        counter = 0
        return self.find_best_move(tboard, validMoves)

    def find_best_move(self, tboard, validMoves):
        global nextMove
        nextMove = None
        # self.negamax_ab(tboard, validMoves, DEPTH, -1000, 1000, -1)
        self.negamax(tboard, validMoves, DEPTH, -1)
        return nextMove

    def negamax(self, tboard, validMoves, depth, turnMultiplier):
        global nextMove, counter
        counter += 1
        if depth == 0:
            return turnMultiplier * tboard.evaluate()
        maxScore = -1000
        for move in validMoves:
            piece = move.piece
            tboard.move(piece, move)
            nextMoves = tboard.getValidMoves()
            score = -self.negamax(tboard, nextMoves, depth - 1, -turnMultiplier)
            if score > maxScore:
                maxScore = score
                if depth == DEPTH:
                    nextMove = move
            tboard.undo_minimax_move(piece, move)
        return maxScore

    def negamax_ab(self, tboard, validMoves, depth, alpha, beta, turnMultiplier):
        global nextMove, counter
        counter += 1
        if depth == 0:
            return turnMultiplier * tboard.evaluate()
        maxScore = -1000
        for move in validMoves:
            piece = move.piece
            tboard.move(piece, move)
            nextMoves = tboard.getValidMoves()
            score = -self.negamax_ab(tboard, nextMoves, depth - 1, -beta, -alpha, -turnMultiplier)
            if score > maxScore:
                maxScore = score
                if depth == DEPTH:
                    nextMove = move
            tboard.undo_minimax_move(piece, move)
            if maxScore > alpha:
                alpha = maxScore
            if alpha >= beta:
                break
        return maxScore
