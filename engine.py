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

    def negamax_ab(self, tboard, validMoves, depth, alpha, beta, turnMultiplier,     null_move_allowed=True, is_root=False):
      global nextMove, counter
      counter += 1
      
      if depth == 0:
          return turnMultiplier * tboard.evaluate()
  
      # Null move pruning
      R = 2  # Reduction factor
      
      if null_move_allowed and depth >= R + 1 and not tboard.in_check():
          tboard.make_null_move()
          score = -self.negamax_ab(tboard, validMoves, depth - R - 1, -beta, -beta + 1, -turnMultiplier, False, False)
          tboard.undo_null_move()
          
          if score >= beta:
              return score
  
      maxScore = float('-inf')
      first_move = True
      
      for move in validMoves:
          # AEL-Pruning
          if self.ael_pruning(move, depth, alpha):
              continue
  
          # Enhanced Forward Pruning
          if self.efp_pruning(move, depth, alpha, beta):
              continue
  
          piece = move.piece
          tboard.move(piece, move)
          nextMoves = tboard.getValidMoves()
  
          if first_move:
              first_move = False
              score = -self.negamax_ab(tboard, nextMoves, depth - 1, -beta, -alpha, -turnMultiplier, True, False)
          else:
              # PVS search
              score = -self.negamax_ab(tboard, nextMoves, depth - 1, -(alpha + 1), -alpha, -turnMultiplier, True, False)
              
              if alpha < score < beta:
                  score = -self.negamax_ab(tboard, nextMoves, depth - 1, -beta, -alpha, -turnMultiplier, True, False)
  
          tboard.undo_move(piece, move)
  
          if score > maxScore:
              maxScore = score
              
              if is_root:
                  nextMove = move
  
          if maxScore > alpha:
              alpha = maxScore
              
          if alpha >= beta:
              break
  
      return maxScore
  
    def efp_pruning(self, move, depth, alpha, beta):
      if depth > 2 and self.is_less_promising(move):
          margin = (beta - alpha) / 2
          
          if self.ael_pruning(move, depth, alpha + margin):
              return True
          
      return False
    
    def ael_pruning(self, move, depth, alpha):
      # AEL pruning
      if move.promotion or move.captured_piece is not None:
          return False
  
      if depth >= 3:
          if move.piece.piece_type == chess.PAWN:
              return False
          elif move.piece.piece_type == chess.KNIGHT and alpha >= self.knight_values[depth]:
              return True
          elif move.piece.piece_type == chess.BISHOP and alpha >= self.bishop_values[depth]:
              return True
          elif move.piece.piece_type == chess.ROOK and alpha >= self.rook_values[depth]:
              return True
          elif move.piece.piece_type == chess.QUEEN and alpha >= self.queen_values[depth]:
              return True
  
      return False

    
    def is_less_promising(self, move):
      # Implement your heuristic here to determine if a move is less promising.
      # For example, you can consider moves that capture lower-valued pieces as less promising.
      # This is just a simple example and may not be the most effective heuristic.
      return move.captured_piece is not None and move.captured_piece.value < move.piece.value
