from const import *
from square import Square
from piece import *
from move import Move
class Board:
    
    def __init__(self):
        self.squares = [[0,0,0,0,0,0,0,0] for col in range(COLS)]
        self.last_move = None
        self._create()
        self._add_pieces('white')
        self._add_pieces('black')

    def move(self, piece, move):
        initial = move.initial
        final = move.final

        # console board move update
        self.squares[initial.row][initial.col].piece = None
        self.squares[final.row][final.col].piece = piece


        # pawn promotion
        if isinstance(piece, Pawn):
            self.check_promotion(piece, final)

        # move
        piece.moved = True

        # clear valid moves
        piece.clear_moves()

        # set last move
        self.last_move = move

    def valid_move(self, piece, move):
        return move in piece.moves

    def check_promotion(self, piece, final):
        if final.row == 0 or final.row == 7:
            self.squares[final.row][final.col].piece = Queen(piece.color)

    def calc_moves(self, piece, row, col):
        '''
        Calculate all possible (valid) moves for a specific piece
        '''

        def pawn_moves():
            possible_moves = []
            first_move = True if piece.moved == False else False
            major3 = [[row + 1, col + i] for i in range(-1, 2)] if piece.dir > 0 else [[row - 1, col + i] for i in range(-1, 2)]

            if self.squares[row + piece.dir][col].isempty():
                possible_moves.append((row + piece.dir, col))
                if self.squares[row + (2 * piece.dir)][col].isempty() and first_move == True:
                    possible_moves.append((row + (piece.dir * 2), col))

            for move in major3:
                if Square.in_range(move[0], move[1]):
                    if major3.index(move) % 2 == 0:
                        if self.squares[move[0]][move[1]].has_rival_piece(piece.color):
                            possible_moves.append((move[0], move[1]))

            for move in possible_moves:
                initial = Square(row, col)
                final = Square(move[0], move[1])

                move = Move(initial, final)
                piece.add_move(move)
                        

            

        def knight_moves():
            possible_moves = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if i ** 2 + j ** 2 == 5:
                        possible_moves.append((row + i, col + j))

            for move in possible_moves:
                move_row, move_col = move

                if Square.in_range(move_row, move_col):
                    if self.squares[move_row][move_col].isempty_or_rival(piece.color):
                        initial = Square(row, col)
                        final = Square(move_row, move_col) # Piece = Piece
                        
                        move = Move(initial, final)
                        piece.add_move(move)

        def king_moves():
            # 9 possible moves
            possible_moves = []
            for y in range(3):
                for x in range(3):
                    possible_moves.append((row - 1 + y, col - 1 + x))

            for move in possible_moves:
                move_row, move_col = move

                if Square.in_range(move_row, move_col):
                    if self.squares[move_row][move_col].isempty_or_rival(piece.color):
                        initial = Square(row, col)
                        final = Square(move_row, move_col)

                        move = Move(initial, final)
                        piece.add_move(move)
        def bishop_moves():
            possible_moves = [[[row + i, col + i] for i in range(1, 8)],
                         [[row + i, col - i] for i in range(1, 8)],
                         [[row - i, col + i] for i in range(1, 8)],
                         [[row - i, col - i] for i in range(1, 8)]]
            for direction in possible_moves:
                for move in direction:
                    if Square.in_range(move[0], move[1]):
                        if self.squares[move[0]][move[1]].isempty():
                            initial = Square(row, col)
                            final = Square(move[0], move[1])

                            move = Move(initial, final)
                            piece.add_move(move)
                        else:
                            if self.squares[move[0]][move[1]].has_rival_piece(piece.color):
                                initial = Square(row, col)
                                final = Square(move[0], move[1])
                                
                                move = Move(initial, final)
                                piece.add_move(move)
                            break
        def rook_moves():
            possible_moves = [[[row + i, col] for i in range(1, 8 - row)],
                     [[row - i, col] for i in range(1, row + 1)],
                     [[row, col + i] for i in range(1, 8 - col)],
                     [[row, col - i] for i in range(1, col + 1)]]
            for direction in possible_moves:
                for move in direction:
                    if Square.in_range(move[0], move[1]):
                        if self.squares[move[0]][move[1]].isempty():
                            initial = Square(row, col)
                            final = Square(move[0], move[1])

                            move = Move(initial, final)
                            piece.add_move(move)
                        else:
                            if self.squares[move[0]][move[1]].has_rival_piece(piece.color):
                                initial = Square(row, col)
                                final = Square(move[0], move[1])
                                
                                move = Move(initial, final)
                                piece.add_move(move)
                            break
        def queen_moves():
            rook_moves()
            bishop_moves() 

        if isinstance(piece, Pawn):
            pawn_moves()
        elif isinstance(piece, Knight):
            knight_moves()
        elif isinstance(piece, Bishop):
            bishop_moves()
        elif isinstance(piece, Rook):
            rook_moves()
        elif isinstance(piece, Queen):
            queen_moves()
        elif isinstance(piece, King):
            king_moves()

    def _create(self):
        for row in range(ROWS):
            for col in range(COLS):
                self.squares[row][col] = Square(row, col)

    def _add_pieces(self, color):
        row_pawn,row_other = (6, 7) if color == 'white' else (1, 0)
        
        # All Pawns
        for col in range(COLS):
            self.squares[row_pawn][col] = Square(row_pawn, col, Pawn(color))

        # Knights
        self.squares[row_other][1] = Square(row_other, 1, Knight(color))
        self.squares[row_other][6] = Square(row_other, 6, Knight(color))
        
        #Bishops
        self.squares[row_other][2] = Square(row_other, 2, Bishop(color))
        self.squares[row_other][5] = Square(row_other, 5, Bishop(color))

        #Rooks
        self.squares[row_other][0] = Square(row_other, 0, Rook(color))
        self.squares[row_other][7] = Square(row_other, 7, Rook(color))

        #Queen
        self.squares[row_other][3] = Square(row_other, 3, Queen(color))

        #King
        self.squares[row_other][4] = Square(row_other, 4, King(color))