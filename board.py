from const import *
from square import Square
from piece import *
from move import Move
from sound import Sound
import chess
import copy
import os

class Board:

    def __init__(self, tboard):
        self.squares = [[0, 0, 0, 0, 0, 0, 0, 0] for col in range(COLS)]
        self.last_move = None
        self.move_list = []
        self.tboard = tboard
        self.next_player = 'white'
        self._create()
        self._add_pieces('white')
        self._add_pieces('black')
        self.calc_moves(self.tboard)
        self.eval = 0

    def move(self, piece, move, testing=False):
        initial = move.initial
        final = move.final

        en_passant_empty = self.squares[final.row][final.col].isempty()

        # console board move update
        self.squares[initial.row][initial.col].piece = None
        self.squares[final.row][final.col].piece = piece

        if isinstance(piece, Pawn):
            # en passant capture
            diff = final.col - initial.col
            if diff != 0 and en_passant_empty:
                # console board move update
                self.squares[initial.row][initial.col + diff].piece = None
                self.squares[final.row][final.col].piece = piece
                if not testing:
                    sound = Sound(
                        os.path.join('assets/sounds/capture.wav'))
                    sound.play()
            
            # pawn promotion
            else:
                self.check_promotion(piece, final)

        # king castling
        if isinstance(piece, King):
            if self.castling(initial, final) and not testing:
                diff = final.col - initial.col
                print(diff)
                print(piece.name)
                rook = piece.left_rook if (diff < 0) else piece.right_rook
                if diff < 0:
                    self.move(piece.left_rook, piece.left_rook.moves[-3], testing=True)
                else:
                    self.move(piece.right_rook, piece.right_rook.moves[-1], testing=True)

        # move
        piece.moved = True

        # clear valid moves
        self.clear_all_moves()

        # update player turn
        if not testing:
            self.next_player = 'black' if self.next_player == 'white' else 'white'

        # set last move
        self.last_move = move
        self.move_list.append(move)

    def clear_all_moves(self):
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    self.squares[row][col].piece.clear_moves()
    def get_piece_from_move(self, move):
        if self.squares[move.final.row][move.final.col].piece:
            return self.squares[move.final.row][move.final.col].piece
    def undo_move(self):
        initial = self.last_move.initial
        final = self.last_move.final
        piece = self.squares[final.row][final.col].piece
        taken_piece = final.piece
        self.squares[initial.row][initial.col].piece = piece
        self.squares[final.row][final.col].piece = taken_piece
        self.last_move = self.move_list[len(self.move_list) - 1]
        self.move_list.remove(self.last_move)
        piece.moved = False
    def valid_move(self, piece, move):
        return move in piece.moves

    def check_promotion(self, piece, final):
        if final.row == 0 or final.row == 7:
            self.squares[final.row][final.col].piece = Queen(piece.color)

    def castling(self, initial, final):
        return abs(initial.col - final.col) == 2

    def find_piece_in_square(self, square):
        if square.piece:
            return self.piece
    def will_take_enemy_piece(self, move, color):
        if self.squares[move.final.row][move.final.col].has_enemy_piece(color):
            return True
        else:
            return False
    def convert_move(self, move):
        # This function converts a python-chess move into a move our code can understand
        ascii_move_initial = chess.square_name(move.from_square)
        ascii_move_final = chess.square_name(move.to_square)
        initial = Square(INV_SQUARE_NAMES.get(ascii_move_initial)[0], INV_SQUARE_NAMES.get(ascii_move_initial)[1])
        final = Square(INV_SQUARE_NAMES.get(ascii_move_final)[0], INV_SQUARE_NAMES.get(ascii_move_final)[1])
        piece = self.squares[initial.row][initial.col].piece
        move = Move(initial, final, piece)
        return move
    def show_available_moves(self, move):
        pass
    def set_true_en_passant(self, piece):
        
        if not isinstance(piece, Pawn):
            return

        for row in range(ROWS):
            for col in range(COLS):
                if isinstance(self.squares[row][col].piece, Pawn):
                    self.squares[row][col].piece.en_passant = False
        
        piece.en_passant = True
    def getValidMoves(self):
        move_list = []
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    if piece.color == self.next_player:
                        self.calc_moves(piece, row, col)
                        for move in piece.moves:
                            move_list.append(move)
        return move_list

    def in_check(self, piece, move):
        temp_piece = copy.deepcopy(piece)
        temp_board = copy.deepcopy(self)
        temp_board.move(temp_piece, move, testing=True)
        
        for row in range(ROWS):
            for col in range(COLS):
                if temp_board.squares[row][col].has_enemy_piece(piece.color):
                    p = temp_board.squares[row][col].piece
                    temp_board.calc_moves(p, row, col, bool=False)
                    for m in p.moves:
                        if isinstance(m.final.piece, King):
                            return True
        
        return False

    # AI functions
    def evaluate(self):
        eval = 0
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    eval += piece.value
        return round(eval, 1) if eval != -0.0 else 0
    def undo_minimax_move(self, piece, move):
        initial = move.initial
        final = move.final
        taken_piece = final.piece
        self.squares[initial.row][initial.col].piece = piece
        self.squares[final.row][final.col].piece = taken_piece
        piece.moved = False
    def calc_weight(self, piece, move):
        if self.will_take_enemy_piece(move, piece.color):
            taken_piece = self.get_piece_from_move(move)
            return taken_piece.value
        else:
            return 0


    def calc_moves(self, tboard):
        for move in tboard.legal_moves:
            new_move = self.convert_move(move)
            piece = new_move.piece
            if piece:
                piece.moves.append(new_move)

    def _create(self):
        for row in range(ROWS):
            for col in range(COLS):
                self.squares[row][col] = Square(row, col)

    def _add_pieces(self, color):
        row_pawn, row_other = (6, 7) if color == 'white' else (1, 0)

        # pawns
        for col in range(COLS):
            self.squares[row_pawn][col] = Square(row_pawn, col, Pawn(color))

        # knights
        self.squares[row_other][1] = Square(row_other, 1, Knight(color))
        self.squares[row_other][6] = Square(row_other, 6, Knight(color))

        # bishops
        self.squares[row_other][2] = Square(row_other, 2, Bishop(color))
        self.squares[row_other][5] = Square(row_other, 5, Bishop(color))

        # rooks
        self.squares[row_other][0] = Square(row_other, 0, Rook(color))
        self.squares[row_other][7] = Square(row_other, 7, Rook(color))

        # queen
        self.squares[row_other][3] = Square(row_other, 3, Queen(color))

        # king
        self.squares[row_other][4] = Square(row_other, 4, King(color))
        self.squares[row_other][4].piece.left_rook = self.squares[row_other][0].piece
        self.squares[row_other][4].piece.right_rook = self.squares[row_other][7].piece
        