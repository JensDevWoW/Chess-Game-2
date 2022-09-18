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
        tboard = copy.deepcopy(self.board)
        validMoves = self.board.getValidMoves('black')
        random.shuffle(validMoves)
        return self.find_best_move(tboard, validMoves)

    def find_best_move(self, tboard, validMoves):
        global nextMove
        nextMove = None
        self.negamax(tboard, validMoves, DEPTH, -1)
        return nextMove 

    def minimax_scoring(self, tboard, validMoves, depth, whiteToMove):
        global nextMove
        if depth == 0:
            return tboard.evaluate()

        if whiteToMove:
            maxScore = -10000
            for move in validMoves:
                piece = tboard.squares[move.initial.row][move.initial.col].piece
                tboard.move(piece, move)
                nextMoves = tboard.getValidMoves('black')
                score = self.minimax_scoring(tboard, nextMoves, depth - 1, False)
                if score > maxScore:
                    maxScore = score
                    if depth == DEPTH:
                        nextMove = move
                tboard.undo_minimax_move(piece, move)
            return maxScore
        else:
            minScore = 10000
            for move in validMoves:
                piece = tboard.squares[move.initial.row][move.initial.col].piece
                tboard.move(piece, move)
                nextMoves = tboard.getValidMoves('white')
                score = self.minimax_scoring(tboard, nextMoves, depth - 1, True)
                if score < minScore:
                    minScore = score
                    if depth == DEPTH:
                        nextMove = move
                tboard.undo_minimax_move(piece, move)
            return minScore
    def negamax(self, tboard, validMoves, depth, turnMultiplier):
        global nextMove
        if depth == 0:
            return turnMultiplier * tboard.evaluate()
        maxScore = -1000
        for move in validMoves:
            piece = tboard.squares[move.initial.row][move.initial.col].piece
            tboard.move(piece, move)
            turn = 'black' if turnMultiplier < 0 else 'white'
            nextMoves = tboard.getValidMoves(turn)
            score = -self.negamax(tboard, nextMoves, depth-1, -turnMultiplier)
            if score > maxScore:
                maxScore = score
                if depth == DEPTH:
                    nextMove = move
            tboard.undo_minimax_move(piece, move)
        return maxScore
                    
    def negamax_ab(self, tboard, validMoves, depth, alpha, beta, turnMultiplier):
        global nextMove
        if depth == 0:
            return turnMultiplier * tboard.evaluate()

        maxScore = -1000
        for move in validMoves:
            piece = tboard.squares[move.initial.row][move.initial.col].piece
            tboard.move(piece, move)
            turn = 'black' if turnMultiplier < 0 else 'white'
            nextMoves = tboard.getValidMoves(turn)
            score = -self.negamax_ab(tboard, nextMoves, depth-1, -beta, -alpha, -turnMultiplier)
            if score > maxScore:
                maxScore = score
                if depth == DEPTH:
                    nextMove = move
            tboard.undo_minimax_move(piece, move)
            if maxScore > alpha: # pruning
                alpha = maxScore
            if alpha >= beta:
                break
        return maxScore           

            