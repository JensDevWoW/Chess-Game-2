from const import *
from square import Square
from piece import *
from move import Move
from sound import Sound
import copy
import os
from zobrist import ZOBRIST_KEYS
import random

class Board:

    def __init__(self):
        self.squares = [[0, 0, 0, 0, 0, 0, 0, 0] for col in range(COLS)]
        self.last_move = None
        self.move_list = []
        self.next_player = 'white'
        self._create()
        self._add_pieces('white')
        self._add_pieces('black')
        self.eval = 0
        self.current_hash = self.get_hash()  # Initialize the board's current hash

    def get_hash(self):
        """Generates a unique hash for the current board position using Zobrist hashing."""
        board_hash = 0

        # Iterate over each square and XOR the Zobrist keys for pieces on the board
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    color = piece.color
                    piece_type = piece.name
                    # XOR the hash with the Zobrist key for this piece on this square
                    board_hash ^= ZOBRIST_KEYS[color][piece_type][row][col]

        # Optionally include the player's turn in the hash
        if self.next_player == 'white':
            board_hash ^= random.getrandbits(64)  # Random value for the turn (white to move)

        return board_hash

    def move(self, piece, move, testing=False):
        initial = move.initial
        final = move.final

        # Capture the piece at the final square if it exists
        move.captured_piece = self.squares[final.row][final.col].piece

        en_passant_empty = self.squares[final.row][final.col].isempty()

        # Update the board state
        self.squares[initial.row][initial.col].piece = None
        self.squares[final.row][final.col].piece = piece

        if isinstance(piece, Pawn):
            # Handle en passant capture
            diff = final.col - initial.col
            if diff != 0 and en_passant_empty:
                self.squares[initial.row][initial.col + diff].piece = None
                self.squares[final.row][final.col].piece = piece
                if not testing:
                    sound = Sound(os.path.join('assets/sounds/capture.wav'))
                    sound.play()
            else:
                self.check_promotion(piece, final)

        # Handle castling for King
        if isinstance(piece, King):
            if self.castling(initial, final) and not testing:
                diff = final.col - initial.col
                rook = piece.left_rook if (diff < 0) else piece.right_rook
                self.move(rook, rook.moves[-1])

        # Mark piece as moved and clear valid moves
        piece.moved = True
        piece.clear_moves()

        # Update turn
        self.next_player = 'black' if self.next_player == 'white' else 'white'

        # Update last move
        self.last_move = move
        self.move_list.append(move)


    def is_rook_on_open_file(self, row, col):
        """Check if a rook is on an open or semi-open file."""
        open_file = True
        semi_open_file = True

        for r in range(ROWS):
            if r != row:
                if self.squares[r][col].has_piece():
                    piece = self.squares[r][col].piece
                    # If a piece of the same color is blocking, it's not an open file
                    if piece.color == self.squares[row][col].piece.color:
                        open_file = False
                        semi_open_file = False
                        break
                    # If an opponent's pawn is blocking, it's only a semi-open file
                    if isinstance(piece, Pawn):
                        open_file = False

        return open_file, semi_open_file


    def is_king_in_check(self, color):
        """Check if the king of the specified color is in check."""
        king_position = None

        # Find the king's position on the board
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    if isinstance(piece, King) and piece.color == color:
                        king_position = (row, col)
                        break
            if king_position:
                break

        # If king is not found (should not happen in normal chess rules)
        if not king_position:
            return False

        # Check if any opposing piece can move to the king's position
        opponent_color = 'black' if color == 'white' else 'white'
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    if piece.color == opponent_color:
                        self.calc_moves(piece, row, col, bool=False)
                        for move in piece.moves:
                            if move.final.row == king_position[0] and move.final.col == king_position[1]:
                                return True

        return False


    def clear_all_moves(self):
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    self.squares[row][col].piece.clear_moves()
    def get_piece_from_move(self, move):
        if self.squares[move.final.row][move.final.col].piece:
            return self.squares[move.final.row][move.final.col].piece
    def undo_move(self, piece, move):
        """Undo the last move made on the board."""
        initial = move.initial
        final = move.final

        # Restore the piece to its initial position
        self.squares[initial.row][initial.col].piece = piece

        # Restore any captured piece to its original square
        self.squares[final.row][final.col].piece = move.captured_piece

        # Reset the moved status of the piece
        piece.moved = False

        # Update the player turn
        self.next_player = 'black' if self.next_player == 'white' else 'white'

        # Remove the last move from the move list
        self.move_list.pop()

        # Update the board's last move
        self.last_move = self.move_list[-1] if self.move_list else None
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

    def evaluate(self):
        """Evaluate the current board position with a comprehensive evaluation function."""
        eval = 0

        # Factors to consider
        piece_activity = 0
        king_safety = 0
        pawn_structure = 0

        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    # Strong emphasis on material value
                    eval += piece.value

                    # Piece activity and position-specific bonuses
                    piece_activity += self.positional_bonus(piece, row, col)

                    # Consider king safety especially during mid and endgames
                    if isinstance(piece, King):
                        king_safety += self.evaluate_king_safety(piece, row, col)

                    # Evaluate pawn structure for doubled, isolated, or backward pawns
                    if isinstance(piece, Pawn):
                        pawn_structure += self.evaluate_pawn_structure(piece, row, col)

        # Combine factors into the final evaluation score
        eval += piece_activity + king_safety + pawn_structure

        return round(eval, 1) if eval != -0.0 else 0

    def positional_bonus(self, piece, row, col):
        """Evaluate positional bonuses for pieces based on their position and activity."""
        central_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        near_central_squares = [(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5), (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5, 5)]
        bonus = 0

        # General bonuses for central control
        if (row, col) in central_squares:
            bonus += 2 if piece.color == 'white' else -2
        elif (row, col) in near_central_squares:
            bonus += 1 if piece.color == 'white' else -1

        # Piece-specific bonuses
        if isinstance(piece, Knight):
            bonus += self.knight_position_bonus(row, col)
        elif isinstance(piece, Bishop):
            bonus += self.bishop_position_bonus(row, col)
        elif isinstance(piece, Rook):
            bonus += self.rook_position_bonus(row, col)
        elif isinstance(piece, Queen):
            bonus += self.queen_position_bonus(row, col)
        elif isinstance(piece, Pawn):
            bonus += self.pawn_position_bonus(row, col)
        elif isinstance(piece, King):
            bonus += self.king_position_bonus(row, col)

        return bonus

    def is_doubled_pawn(self, pawn, col):
        """Check if the given pawn is doubled on its file."""
        pawn_count = 0
        for row in range(ROWS):
            if self.squares[row][col].has_piece() and isinstance(self.squares[row][col].piece, Pawn):
                if self.squares[row][col].piece.color == pawn.color:
                    pawn_count += 1
        return pawn_count > 1

    def is_isolated_pawn(self, pawn, col):
        """Check if the given pawn is isolated (no pawns on adjacent files)."""
        left_file = col - 1
        right_file = col + 1

        isolated = True
        if left_file >= 0:
            for row in range(ROWS):
                if self.squares[row][left_file].has_piece() and isinstance(self.squares[row][left_file].piece, Pawn):
                    if self.squares[row][left_file].piece.color == pawn.color:
                        isolated = False
                        break

        if right_file < COLS:
            for row in range(ROWS):
                if self.squares[row][right_file].has_piece() and isinstance(self.squares[row][right_file].piece, Pawn):
                    if self.squares[row][right_file].piece.color == pawn.color:
                        isolated = False
                        break

        return isolated


    def is_square_threatened(self, row, col, opponent_color):
        """
        Check if the square at (row, col) is threatened by any piece of the opponent's color.
        """
        # Iterate over all squares and check moves of opponent pieces
        for r in range(ROWS):
            for c in range(COLS):
                if self.squares[r][c].has_piece():
                    piece = self.squares[r][c].piece
                    if piece.color == opponent_color:
                        self.calc_moves(piece, r, c, bool=False)  # Calculate possible moves without checks
                        for move in piece.moves:
                            # Check if any move targets the specified square
                            if move.final.row == row and move.final.col == col:
                                return True
        return False


    def is_backward_pawn(self, pawn, row, col):
        """Check if the given pawn is backward (cannot advance safely and lacks support)."""
        direction = -1 if pawn.color == 'white' else 1
        behind_row = row + direction

        if behind_row >= 0 and behind_row < ROWS:
            if self.squares[behind_row][col].has_piece() and isinstance(self.squares[behind_row][col].piece, Pawn):
                if self.squares[behind_row][col].piece.color != pawn.color and self.is_isolated_pawn(pawn, col):
                    return True

        return False


    def evaluate_king_safety(self, king, row, col):
        """Evaluate king safety based on position, pawn cover, and opponent threats."""
        safety = 0
        # Check for pawn cover and whether the king is castled
        if self.is_king_castled(king, row, col):
            safety += 3
        # Penalties for exposed or poorly defended kings
        if self.is_king_exposed(king, row, col):
            safety -= 3
        return safety

    def evaluate_pawn_structure(self, pawn, row, col):
        """Evaluate pawn structure, penalizing weaknesses such as doubled, isolated, or backward pawns."""
        structure_score = 0
        # Penalize doubled pawns
        if self.is_doubled_pawn(pawn, col):
            structure_score -= 1
        # Penalize isolated pawns with no adjacent pawn support
        if self.is_isolated_pawn(pawn, col):
            structure_score -= 1.5
        # Penalize backward pawns that cannot advance easily
        if self.is_backward_pawn(pawn, row, col):
            structure_score -= 1
        return structure_score


    def knight_position_bonus(self, row, col):
        """Evaluate positional bonuses and penalties specific to knights."""
        bonus = 0
        # Encourage centralization of knights
        if (col in [2, 3, 4, 5]) and (row in [2, 3, 4, 5]):
            bonus += 2
        # Penalize knights on the edge of the board
        if col == 0 or col == 7 or row == 0 or row == 7:
            bonus -= 2
        return bonus

    def bishop_position_bonus(self, row, col):
        """Evaluate positional bonuses specific to bishops."""
        bonus = 0
        # Reward active bishops and those that are not blocked by their own pawns
        if not self.is_bishop_blocked(row, col):
            bonus += 1.5
        # Diagonal control and mobility
        if (col in [2, 5]) and (row in [2, 5]):
            bonus += 1
        return bonus

    # Add this method inside your Board class

    def is_king_castled(self, king, row, col):
        """Check if the king is in a typical castling position."""
        # Common castling squares for kings
        castling_positions = [(7, 6), (7, 2), (0, 6), (0, 2)]
        return (row, col) in castling_positions


    def rook_position_bonus(self, row, col):
        """Evaluate positional bonuses specific to rooks."""
        bonus = 0
        open_file, semi_open_file = self.is_rook_on_open_file(row, col)
        # Rooks on open files are very valuable
        if open_file:
            bonus += 2
        elif semi_open_file:
            bonus += 1
        # Penalty for rooks that haven't been activated (still on original squares)
        if (row == 0 and col in [0, 7]) or (row == 7 and col in [0, 7]):
            bonus -= 1
        return bonus


    def queen_position_bonus(self, row, col):
        """Evaluate positional bonuses specific to queens."""
        bonus = 0
        # Encourage early queen activity but discourage early overexposure
        if row in [3, 4] and col in [3, 4]:
            bonus += 1.5
        # Discourage queens from being trapped or overexposed early
        if (row, col) in [(0, 3), (7, 3), (0, 4), (7, 4)]:
            bonus -= 1
        return bonus

    def pawn_position_bonus(self, row, col):
        """Evaluate positional bonuses specific to pawns."""
        bonus = 0
        # Reward pawns that control center squares
        if (row, col) in [(3, 3), (3, 4), (4, 3), (4, 4)]:
            bonus += 0.5
        # Penalize doubled, isolated, or backward pawns handled separately
        return bonus

    def king_position_bonus(self, row, col):
        """Evaluate positional bonuses specific to kings, emphasizing safety."""
        bonus = 0
        # Early game: King safety in castling position is critical
        if (row, col) in [(7, 6), (7, 2), (0, 6), (0, 2)]:
            bonus += 3
        # Encourage castling and king safety in general
        if (row, col) in [(0, 4), (7, 4)]:  # Discourage center position if not castled
            bonus -= 2
        return bonus

    # Add this method inside your Board class

    def is_king_exposed(self, king, row, col):
        """Check if the king is exposed or lacks sufficient pawn cover."""
        # Define directions to check surrounding squares for pawn cover
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        safety = 0

        # Check each surrounding square for protection by friendly pawns
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < ROWS and 0 <= c < COLS:
                square = self.squares[r][c]
                if square.has_piece() and isinstance(square.piece, Pawn) and square.piece.color == king.color:
                    safety += 1

        # Consider the king exposed if it has fewer than 2 pawns protecting it
        return safety < 2


    def evaluate_king_safety(self, king, row, col):
        """Evaluate king safety based on position, pawn cover, and opponent threats."""
        safety = 0
        # Check for pawn cover and whether the king is castled
        if self.is_king_castled(king, row, col):
            safety += 3
        # Penalties for exposed or poorly defended kings
        if self.is_king_exposed(king, row, col):
            safety -= 3
        return safety


    def evaluate_pawn_structure(self, pawn, row, col):
        """Evaluate pawn structure, penalizing weaknesses such as doubled, isolated, or backward pawns."""
        structure_score = 0
        # Penalize doubled pawns
        if self.is_doubled_pawn(pawn, col):
            structure_score -= 1
        # Penalize isolated pawns with no adjacent pawn support
        if self.is_isolated_pawn(pawn, col):
            structure_score -= 1.5
        # Penalize backward pawns that cannot advance easily
        if self.is_backward_pawn(pawn, row, col):
            structure_score -= 1
        return structure_score

    def is_bishop_blocked(self, row, col):
        """Check if a bishop is blocked by its own pawns."""
        piece = self.squares[row][col].piece
        if isinstance(piece, Bishop):
            # Check if pawns of the same color block the bishop's diagonal paths
            # Simplified example: you may need a more complex implementation
            return (row > 0 and self.squares[row - 1][col].has_piece() and isinstance(self.squares[row - 1][col].piece, Pawn)) or \
                (row < ROWS - 1 and self.squares[row + 1][col].has_piece() and isinstance(self.squares[row + 1][col].piece, Pawn))
        return False



    def is_rook_active(self, rook, row, col):
        """Check if a rook is on an open or semi-open file, making it active."""
        # Check if the rook's column has no pawns or only opposing pawns
        for r in range(ROWS):
            if self.squares[r][col].has_piece() and isinstance(self.squares[r][col].piece, Pawn):
                if self.squares[r][col].piece.color == rook.color:
                    return False
        return True



    
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


    def calc_moves(self, piece, row, col, bool=True):
        '''
            Calculate all the possible (valid) moves of an specific piece on a specific position
        '''
        
        def pawn_moves():
            # steps
            steps = 1 if piece.moved else 2

            # vertical moves
            start = row + piece.dir
            end = row + (piece.dir * (1 + steps))
            for possible_move_row in range(start, end, piece.dir):
                if Square.in_range(possible_move_row):
                    if self.squares[possible_move_row][col].isempty():
                        # create initial and final move squares
                        initial = Square(row, col)
                        final = Square(possible_move_row, col)
                        # create a new move
                        move = Move(initial, final, piece)

                        # check potencial checks
                        if bool:
                            if not self.in_check(piece, move):
                                # append new move
                                piece.add_move(move)
                                move.weight = self.calc_weight(piece, move)
                        else:
                            # append new move
                            piece.add_move(move)
                            move.weight = self.calc_weight(piece, move)
                    # blocked
                    else: break
                # not in range
                else: break

            # diagonal moves
            possible_move_row = row + piece.dir
            possible_move_cols = [col-1, col+1]
            for possible_move_col in possible_move_cols:
                if Square.in_range(possible_move_row, possible_move_col):
                    if self.squares[possible_move_row][possible_move_col].has_enemy_piece(piece.color):
                        # create initial and final move squares
                        initial = Square(row, col)
                        final_piece = self.squares[possible_move_row][possible_move_col].piece
                        final = Square(possible_move_row, possible_move_col, final_piece)
                        final.piece = final_piece
                        # create a new move
                        move = Move(initial, final, piece)
                        
                        # check potencial checks
                        if bool:
                            if not self.in_check(piece, move):
                                # append new move
                                piece.add_move(move)
                                move.weight = self.calc_weight(piece, move)
                        else:
                            # append new move
                            piece.add_move(move)
                            move.weight = self.calc_weight(piece, move)

            # en passant moves
            r = 3 if piece.color == 'white' else 4
            fr = 2 if piece.color == 'white' else 5
            # left en pessant
            if Square.in_range(col-1) and row == r:
                if self.squares[row][col-1].has_enemy_piece(piece.color):
                    p = self.squares[row][col-1].piece
                    if isinstance(p, Pawn):
                        if p.en_passant:
                            # create initial and final move squares
                            initial = Square(row, col)
                            final = Square(fr, col-1, p)
                            # create a new move
                            move = Move(initial, final, piece)
                            
                            # check potencial checks
                            if bool:
                                if not self.in_check(piece, move):
                                    # append new move
                                    piece.add_move(move)
                                    move.weight = self.calc_weight(piece, move)
                            else:
                                # append new move
                                piece.add_move(move)
                                move.weight = self.calc_weight(piece, move)
            
            # right en pessant
            if Square.in_range(col+1) and row == r:
                if self.squares[row][col+1].has_enemy_piece(piece.color):
                    p = self.squares[row][col+1].piece
                    if isinstance(p, Pawn):
                        if p.en_passant:
                            # create initial and final move squares
                            initial = Square(row, col)
                            final = Square(fr, col+1, p)
                            # create a new move
                            move = Move(initial, final, piece)
                            
                            # check potencial checks
                            if bool:
                                if not self.in_check(piece, move):
                                    # append new move
                                    piece.add_move(move)
                                    move.weight = self.calc_weight(piece, move)
                            else:
                                # append new move
                                piece.add_move(move)
                                move.weight = self.calc_weight(piece, move)


        def knight_moves():
            # 8 possible moves
            possible_moves = [
                (row-2, col+1),
                (row-1, col+2),
                (row+1, col+2),
                (row+2, col+1),
                (row+2, col-1),
                (row+1, col-2),
                (row-1, col-2),
                (row-2, col-1),
            ]

            for possible_move in possible_moves:
                possible_move_row, possible_move_col = possible_move

                if Square.in_range(possible_move_row, possible_move_col):
                    if self.squares[possible_move_row][possible_move_col].isempty_or_enemy(piece.color):
                        # create squares of the new move
                        initial = Square(row, col)
                        final_piece = self.squares[possible_move_row][possible_move_col].piece
                        final = Square(possible_move_row, possible_move_col, final_piece)
                        final.piece = final_piece
                        # create new move
                        move = Move(initial, final, piece)
                        
                        # check potencial checks
                        if bool:
                            if not self.in_check(piece, move):
                                # append new move
                                piece.add_move(move)
                                move.weight = self.calc_weight(piece, move)
                            else: break
                        else:
                            # append new move
                            piece.add_move(move)
                            move.weight = self.calc_weight(piece, move)

        def straightline_moves(incrs):
            for incr in incrs:
                row_incr, col_incr = incr
                possible_move_row = row + row_incr
                possible_move_col = col + col_incr

                while True:
                    if Square.in_range(possible_move_row, possible_move_col):
                        # create squares of the possible new move
                        initial = Square(row, col)
                        final_piece = self.squares[possible_move_row][possible_move_col].piece
                        final = Square(possible_move_row, possible_move_col, final_piece)
                        final.piece = final_piece
                        # create a possible new move
                        move = Move(initial, final, piece)

                        # empty = continue looping
                        if self.squares[possible_move_row][possible_move_col].isempty():
                            # check potencial checks
                            if bool:
                                if not self.in_check(piece, move):
                                    # append new move
                                    piece.add_move(move)
                                    move.weight = self.calc_weight(piece, move)
                            else:
                                # append new move
                                piece.add_move(move)
                                move.weight = self.calc_weight(piece, move)

                        # has enemy piece = add move + break
                        elif self.squares[possible_move_row][possible_move_col].has_enemy_piece(piece.color):
                            # check potencial checks
                            if bool:
                                if not self.in_check(piece, move):
                                    # append new move
                                    piece.add_move(move)
                                    move.weight = self.calc_weight(piece, move)
                            else:
                                # append new move
                                piece.add_move(move)
                                move.weight = self.calc_weight(piece, move)
                            break

                        # has team piece = break
                        elif self.squares[possible_move_row][possible_move_col].has_team_piece(piece.color):
                            break
                    
                    # not in range
                    else: break

                    # incrementing incrs
                    possible_move_row = possible_move_row + row_incr
                    possible_move_col = possible_move_col + col_incr

        def king_moves():
            adjs = [
                (row-1, col+0), # up
                (row-1, col+1), # up-right
                (row+0, col+1), # right
                (row+1, col+1), # down-right
                (row+1, col+0), # down
                (row+1, col-1), # down-left
                (row+0, col-1), # left
                (row-1, col-1), # up-left
            ]

            # normal moves
            for possible_move in adjs:
                possible_move_row, possible_move_col = possible_move

                if Square.in_range(possible_move_row, possible_move_col):
                    if self.squares[possible_move_row][possible_move_col].isempty_or_enemy(piece.color):
                        # create squares of the new move
                        initial = Square(row, col)
                        final = Square(possible_move_row, possible_move_col) # piece=piece
                        final.piece = self.squares[possible_move_row][possible_move_col].piece
                        # create new move
                        move = Move(initial, final, piece)
                        # check potencial checks
                        if bool:
                            if not self.in_check(piece, move):
                                # append new move
                                piece.add_move(move)
                                move.weight = self.calc_weight(piece, move)
                            else: break
                        else:
                            # append new move
                            piece.add_move(move)
                            move.weight = self.calc_weight(piece, move)

            # castling moves
            if not piece.moved:
                # queen castling
                left_rook = self.squares[row][0].piece
                if isinstance(left_rook, Rook):
                    if not left_rook.moved:
                        for c in range(1, 4):
                            # castling is not possible because there are pieces in between ?
                            if self.squares[row][c].has_piece():
                                break

                            if c == 3:
                                # adds left rook to king
                                piece.left_rook = left_rook

                                # rook move
                                initial = Square(row, 0)
                                final = Square(row, 3)
                                moveR = Move(initial, final, piece)

                                # king move
                                initial = Square(row, col)
                                final = Square(row, 2)
                                moveK = Move(initial, final, piece)

                                # check potencial checks
                                if bool:
                                    if not self.in_check(piece, moveK) and not self.in_check(left_rook, moveR):
                                        # append new move to rook
                                        left_rook.add_move(moveR)
                                        # append new move to king
                                        piece.add_move(moveK)
                                else:
                                    # append new move to rook
                                    left_rook.add_move(moveR)
                                    # append new move king
                                    piece.add_move(moveK)

                # king castling
                right_rook = self.squares[row][7].piece
                if isinstance(right_rook, Rook):
                    if not right_rook.moved:
                        for c in range(5, 7):
                            # castling is not possible because there are pieces in between ?
                            if self.squares[row][c].has_piece():
                                break

                            if c == 6:
                                # adds right rook to king
                                piece.right_rook = right_rook

                                # rook move
                                initial = Square(row, 7)
                                final = Square(row, 5)
                                moveR = Move(initial, final, piece)

                                # king move
                                initial = Square(row, col)
                                final = Square(row, 6)
                                moveK = Move(initial, final, piece)

                                # check potencial checks
                                if bool:
                                    if not self.in_check(piece, moveK) and not self.in_check(right_rook, moveR):
                                        # append new move to rook
                                        right_rook.add_move(moveR)
                                        # append new move to king
                                        piece.add_move(moveK)
                                else:
                                    # append new move to rook
                                    right_rook.add_move(moveR)
                                    # append new move king
                                    piece.add_move(moveK)

        if isinstance(piece, Pawn): 
            pawn_moves()

        elif isinstance(piece, Knight): 
            knight_moves()

        elif isinstance(piece, Bishop): 
            straightline_moves([
                (-1, 1), # up-right
                (-1, -1), # up-left
                (1, 1), # down-right
                (1, -1), # down-left
            ])

        elif isinstance(piece, Rook): 
            straightline_moves([
                (-1, 0), # up
                (0, 1), # right
                (1, 0), # down
                (0, -1), # left
            ])

        elif isinstance(piece, Queen): 
            straightline_moves([
                (-1, 1), # up-right
                (-1, -1), # up-left
                (1, 1), # down-right
                (1, -1), # down-left
                (-1, 0), # up
                (0, 1), # right
                (1, 0), # down
                (0, -1) # left
            ])

        elif isinstance(piece, King): 
            king_moves()

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
        