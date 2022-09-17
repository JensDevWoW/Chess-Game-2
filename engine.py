from board import *
import random
import time
from game import *
from move import *

class Engine:

    def __init__(self):
        self.my_turn = False
        self.color = 'black'
        self.good_moves = []
        self.bad_moves = []
        
    def move(self, board, game):
        move = self.select_move()
        piece = board.squares[move.initial.row][move.initial.col].piece
        board.move(piece, move)
        board.clear_all_moves()
        self.wipe_moves()
        game.next_turn()
    def gen_moves(self, board, game):
        square_list = []
        for row in range(8):
            for col in range(8):
                if board.squares[row][col].has_team_piece('black'):
                    current_square = board.squares[row][col]
                    board.calc_moves(current_square.piece, current_square.row, current_square.col)
                    square_list.append(board.squares[row][col])

        self.sort_moves(square_list)
    def wipe_moves(self):
        self.good_moves = []
        self.bad_moves = []
    def sort_moves(self, squares):
        for square in squares:
            for move in square.piece.moves:
                if move.weight > 0:
                    self.good_moves.append(move)
                else:
                    self.bad_moves.append(move)
    def select_move(self):
        best_move = Move(0, 0)
        if len(self.good_moves) > 0:
            for move in self.good_moves:
                if move.weight > best_move.weight:
                    best_move = move
            print(best_move.weight)
            return best_move
        else:
            rand_num = random.randint(0, len(self.bad_moves) - 1)
            return self.bad_moves[rand_num]  
    
                    
                    

            