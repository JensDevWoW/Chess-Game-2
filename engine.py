from json.encoder import INFINITY
from board import *
import random
import time
from game import *
from move import *
from zobrist import ZOBRIST_KEYS
import json
import copy
from math import log, sqrt
from random import choice

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
        self.transposition_table = {}
        self.max_depth = DEPTH
        self.counter = 0
        self.opening_book = self.load_opening_book()

    def load_opening_book(self):
        try:
            with open('opening_book.json', 'r') as file:
                opening_book = json.load(file)
            print("Opening book loaded successfully.")
            return opening_book
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading opening book: {e}. Continuing without it.")
            return []

    def move(self):
        self.counter = 0
        book_move = self.get_opening_move()
        if book_move:
            print(f"Playing book move: {book_move}")
            self.execute_book_move(book_move)
            return

        valid_moves = self.board.getValidMoves()
        print(f"Valid Moves Found: {len(valid_moves)}")
        
        best_move = self.find_best_move(self.board, valid_moves)
        if best_move is None:
            print("No valid move found!")
            return

        piece = self.board.squares[best_move.initial.row][best_move.initial.col].piece
        print(f"Best Move: {best_move}")
        
        self.board.move(piece, best_move)
        self.board.clear_all_moves()
        self.board.next_player = 'white'
        
        # Print the evaluation score after the move
        eval_score = self.board.evaluate()
        print(f"Evaluation Score after move: {eval_score}")

        print(f"Nodes evaluated: {self.counter}")


    def get_opening_move(self):
        current_fen = self.board_to_fen()
        print(f"Checking opening book for position: {current_fen}")
        
        for entry in self.opening_book:
            if isinstance(entry, dict) and 'fen' in entry and entry['fen'].startswith(current_fen):
                print(f"Opening book match found: {entry['moves']}")
                return self.parse_next_move(entry['moves'])

        print("No matching opening book entry found.")
        return None

    def board_to_fen(self):
        pieces = ""
        empty_count = 0
        
        for row in self.board.squares:
            for square in row:
                if square.has_piece():
                    if empty_count > 0:
                        pieces += str(empty_count)
                        empty_count = 0
                    piece = square.piece
                    pieces += piece.name[0].upper() if piece.color == 'white' else piece.name[0].lower()
                else:
                    empty_count += 1
            if empty_count > 0:
                pieces += str(empty_count)
                empty_count = 0
            pieces += "/"

        pieces = pieces.rstrip('/')
        turn = 'w' if self.board.next_player == 'white' else 'b'
        return f"{pieces} {turn}"

    def parse_next_move(self, moves):
        moves_list = moves.split()
        if len(moves_list) > len(self.board.move_list):
            next_move = moves_list[len(self.board.move_list)]
            return next_move
        return None

    def execute_book_move(self, move):
        initial_col, initial_row, final_col, final_row = move[0], int(move[1]), move[2], int(move[3])
        initial_square = Square(8 - initial_row, ord(initial_col) - ord('a'))
        final_square = Square(8 - final_row, ord(final_col) - ord('a'))
        piece = self.board.squares[initial_square.row][initial_square.col].piece
        move = Move(initial_square, final_square, piece)
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
        valid_moves = self.board.getValidMoves()
        random.shuffle(valid_moves)
        self.counter = 0
        return self.find_best_move(tboard, valid_moves)

    def find_best_move(self, tboard, valid_moves):
        global nextMove
        nextMove = None
        self.negamax_ab(tboard, valid_moves, self.max_depth, float('-inf'), float('inf'), 1, True, True)
        return nextMove

    def negamax_ab(self, tboard, valid_moves, depth, alpha, beta, turnMultiplier, null_move_allowed=True, is_root=False):
        global nextMove
        self.counter += 1

        print(f"Depth: {depth}, Alpha: {alpha}, Beta: {beta}, Turn: {'White' if turnMultiplier == 1 else 'Black'}")

        # Early tactical cutoff: Check for decisive moves
        decisive_move = self.check_for_decisive_move(valid_moves)
        if decisive_move:
            nextMove = decisive_move
            decisive_score = float('inf') if turnMultiplier == 1 else float('-inf')
            print(f"Decisive move found: {decisive_move}, Score: {decisive_score}")
            return decisive_score

        # Transposition Table Lookup
        hash_key = tboard.get_hash()
        if hash_key in self.transposition_table:
            tt_entry = self.transposition_table[hash_key]
            if tt_entry['depth'] >= depth:
                print(f"Using transposition table: {tt_entry}")
                if tt_entry['flag'] == 'EXACT':
                    return tt_entry['score']
                elif tt_entry['flag'] == 'LOWERBOUND' and tt_entry['score'] > alpha:
                    alpha = tt_entry['score']
                elif tt_entry['flag'] == 'UPPERBOUND' and tt_entry['score'] < beta:
                    beta = tt_entry['score']
                if alpha >= beta:
                    return tt_entry['score']

        # Quiescence Search for tactical depth
        if depth == 0:
            quiescence_score = self.quiescence(alpha, beta, tboard, turnMultiplier)
            print(f"Quiescence evaluation at depth 0: {quiescence_score}")
            return quiescence_score

        # Null Move Pruning
        if null_move_allowed and depth >= 3 and not tboard.is_king_in_check(self.board.next_player):
            tboard.make_null_move()
            score = -self.negamax_ab(tboard, valid_moves, depth - 3, -beta, -beta + 1, -turnMultiplier, False, False)
            tboard.undo_null_move()
            if score >= beta:
                print(f"Null move pruning triggered at depth {depth} with score {score}")
                return beta

        ordered_moves = self.order_moves(valid_moves)
        max_score = float('-inf')
        first_move = True

        for move in ordered_moves:
            piece = move.piece

            # Skip moves that endanger pieces without compensation
            if tboard.is_square_threatened(move.final.row, move.final.col, 'white' if piece.color == 'black' else 'black'):
                print(f"Move {move} puts piece in danger: {piece}")
                continue

            tboard.move(piece, move)
            next_moves = tboard.getValidMoves()

            # Additional check after move to ensure safety
            if tboard.is_square_threatened(move.final.row, move.final.col, piece.color):
                print(f"After moving, piece is threatened: {move}")
                tboard.undo_move(piece, move)
                continue

            # Recursively evaluate the move
            if first_move:
                first_move = False
                score = -self.negamax_ab(tboard, next_moves, depth - 1, -beta, -alpha, -turnMultiplier, True, False)
            else:
                score = -self.negamax_ab(tboard, next_moves, depth - 1, -(alpha + 1), -alpha, -turnMultiplier, True, False)
                if alpha < score < beta:
                    score = -self.negamax_ab(tboard, next_moves, depth - 1, -beta, -alpha, -turnMultiplier, True, False)

            tboard.undo_move(piece, move)

            # Log move evaluation for deeper analysis
            print(f"Move: {move}, Score: {score}, Alpha: {alpha}, Beta: {beta}, Max Score: {max_score}")

            # Update max score and alpha for Black or White
            if score > max_score:
                max_score = score
                if is_root:
                    nextMove = move

            # Adjust alpha for maximizing and minimizing player
            alpha = max(alpha, score)
            if alpha >= beta:
                print(f"Alpha-Beta cutoff at move {move}, Score: {score}")
                break

        # Store results in the transposition table
        self.transposition_table[hash_key] = {
            'score': max_score,
            'depth': depth,
            'flag': 'EXACT' if alpha < beta else ('LOWERBOUND' if max_score >= beta else 'UPPERBOUND')
        }

        print(f"Returning max score at depth {depth}: {max_score} for {'White' if turnMultiplier == 1 else 'Black'}")
        return max_score







    def is_immediate_blunder(self, move, tboard):
        """
        Check if making this move would result in an immediate and significant material loss
        without any reasonable compensation, such as sacrificing a queen early.
        """
        piece_value = move.piece.value

        # Temporarily execute the move to evaluate its aftermath
        tboard.move(move.piece, move, testing=True)

        # Determine the opponent's color
        opponent_color = 'black' if move.piece.color == 'white' else 'white'

        # Check if the final position of the move is threatened by the opponent
        if tboard.is_square_threatened(move.final.row, move.final.col, opponent_color):
            # Evaluate if the threatened piece is valuable enough to avoid the move
            if piece_value >= 9:  # Example: Avoid sacrificing high-value pieces like queens
                tboard.undo_move(move.piece, move)
                print(f"Detected blunder: {move} losing a valuable piece!")  # Debug: Print detected blunder
                return True

        # Undo the temporary move after evaluation
        tboard.undo_move(move.piece, move)
        return False





    def print_board_state(self, board):
        """Prints the current board state with a simple visualization."""
        for row in board.squares:
            row_str = ''
            for square in row:
                if square.has_piece():
                    piece = square.piece
                    row_str += f'{piece.name[0]} ' if piece.color == 'white' else f'{piece.name[0].lower()} '
                else:
                    row_str += '. '
            print(row_str)
        print("\n")

    def print_threatened_squares(self, board, color):
        """Visualizes threatened squares on the board by the specified color."""
        threatened = [[0] * COLS for _ in range(ROWS)]

        for row in range(ROWS):
            for col in range(COLS):
                if board.squares[row][col].has_piece() and board.squares[row][col].piece.color == color:
                    piece = board.squares[row][col].piece
                    moves = board.get_threatening_moves(piece, row, col)
                    for move in moves:
                        threatened[move.final.row][move.final.col] = 1

        print(f"Threatened squares by {'white' if color == 'white' else 'black'}:")
        for row in threatened:
            print(' '.join(['X' if x == 1 else '.' for x in row]))
        print("\n")


    def detailed_move_evaluation(self, move):
        """Logs a detailed breakdown of the evaluation for a specific move."""
        score = 0
        print(f"Evaluating Move: {move}")

        # Check for material gain/loss
        if move.captured_piece:
            capture_value = move.captured_piece.value
            print(f"Captured piece value: {capture_value}")
            score += capture_value * 10

        # Check for positional advantages
        central_bonus = self.is_central_move(move)
        print(f"Central control bonus: {central_bonus}")
        score += central_bonus

        # Check for tactical threats
        if self.gives_check(move):
            print("Move gives check")
            score += 50

        print(f"Total evaluation for move: {score}\n")


    def should_check_blunder(self, move, tboard):
        """
        Determine if a move requires a blunder check. Only check for moves involving high-value pieces.
        """
        # Check captures or moves involving high-value pieces
        if move.captured_piece and move.captured_piece.value >= 5:  # Example threshold for valuable captures
            return True
        if isinstance(move.piece, (Queen, Rook)) and tboard.is_square_threatened(move.final, move.piece.color):
            return True
        return False

    def is_safe_move(self, move, tboard):
        """
        Quickly assess if the move leads to an immediate tactical disadvantage.
        """
        piece = move.piece
        tboard.move(piece, move, testing=True)

        # Only evaluate threats to high-value pieces or critical positions
        is_safe = not any(
            threat.captured_piece and threat.captured_piece.value >= piece.value
            for threat in self.get_threats(tboard, move.final)
        )

        tboard.undo_move(piece, move)
        return is_safe

    def get_threats(self, tboard, square):
        """
        Get a list of enemy moves that threaten the given square.
        """
        threats = []
        for move in tboard.getValidMoves():
            if move.final == square and move.captured_piece is not None:
                threats.append(move)
        return threats

    def quiescence(self, alpha, beta, tboard, turnMultiplier):
        stand_pat = turnMultiplier * tboard.evaluate()
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        capture_moves = self.get_capture_moves(tboard)
        for move in capture_moves:
            tboard.move(move.piece, move)
            score = -self.quiescence(-beta, -alpha, tboard, -turnMultiplier)
            tboard.undo_move(move.piece, move)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def get_capture_moves(self, tboard):
        capture_moves = []
        valid_moves = tboard.getValidMoves()

        for move in valid_moves:
            if move.captured_piece is not None:
                capture_moves.append(move)

        return capture_moves

    def order_moves(self, moves):
        def move_score(move):
            score = 0

                # 1. High value for captures with emphasis on capturing valuable pieces
            if move.captured_piece:
                # Double the weight to ensure captures are prioritized highly
                score += move.captured_piece.value * 20

                # Prioritize moves that result in a gain in material (e.g., rook capturing a queen)
                if move.captured_piece.value > move.piece.value:
                    score += 100  # Significantly reward captures of high-value targets

            # Penalty for moves that expose the moved piece to immediate capture without compensation
            if self.is_hanging_piece(move):
                score -= 100  # Heavy penalty for hanging pieces

            # Check if the move is a blunder by losing a valuable piece without compensation
            if self.is_blunder(move):
                score -= 200  # Extra penalty for blunder moves

            if move.promotion:
                score += 90

            if self.gives_check(move):
                score += 50

            if self.is_central_move(move):
                score += 20

            if self.is_development_move(move):
                score += 15

            if isinstance(move.piece, Knight) and (move.final.col in [0, 7] or move.final.row in [0, 7]):
                score -= 20

            if isinstance(move.piece, Rook) and not self.board.is_rook_active(move.piece, move.final.row, move.final.col):
                score -= 10

            if self.blocks_important_piece(move):
                score -= 15

            if isinstance(move.piece, Rook) and self.is_rook_on_open_file(move):
                score += 20

            if self.is_repetitive_move(move):
                score -= 50

            if self.is_castling_move(move):
                score += 30

            if isinstance(move.piece, Pawn) and self.is_bad_pawn_move(move):
                score -= 10

            if self.improves_king_safety(move):
                score += 10

            # Check for tactical cutoff: If a move captures a very high-value piece (like a queen) without trade
            if move.captured_piece and move.captured_piece.value >= 9 and not self.would_lose_piece(move):
                score += 1000  # Arbitrarily high score to ensure it's picked immediately

            return score

        moves.sort(key=move_score, reverse=True)
        return moves

    def is_hanging_piece(self, move):
        """Check if the moved piece is left hanging (vulnerable to capture without adequate defense)."""
        temp_board = copy.deepcopy(self.board)
        temp_board.move(move.piece, move, testing=True)

        # Assume it's the opponent's turn next
        opponent_color = 'white' if move.piece.color == 'black' else 'black'
        opponent_moves = temp_board.getValidMoves()
        for opp_move in opponent_moves:
            # Check if any opponent move captures the moved piece immediately
            if opp_move.captured_piece == move.piece:
                return True
        return False

    def is_blunder(self, move):
        """Check if the move is a blunder (leads to losing material without compensation)."""
        temp_board = copy.deepcopy(self.board)
        temp_board.move(move.piece, move, testing=True)

        # Check if the moved piece or another piece of equal/higher value is immediately at risk
        for opp_move in temp_board.getValidMoves():
            if opp_move.captured_piece and opp_move.captured_piece == move.piece:
                # Penalize if we lose a piece without capturing something of similar value
                return True
        return False


    def would_lose_piece(self, move):
        """Check if the move results in a piece being immediately recaptured."""
        temp_board = copy.deepcopy(self.board)
        temp_board.move(move.piece, move, testing=True)
        # Assume the opponent's turn
        opponent_color = 'white' if move.piece.color == 'black' else 'black'
        # Get all possible opponent moves
        opponent_moves = temp_board.getValidMoves()
        for opp_move in opponent_moves:
            if opp_move.captured_piece and opp_move.captured_piece == move.piece:
                return True
        return False

    def check_for_decisive_move(self, moves):
        """Identify a decisive move if it exists, such as capturing a high-value piece without trade."""
        for move in moves:
            if move.captured_piece and move.captured_piece.value >= 9 and not self.would_lose_piece(move):
                print(f"Decisive move found: {move}")  # Debug: Log the decisive move
                return move
        return None

    def improves_king_safety(self, move):
        piece = move.piece
        if isinstance(piece, King) or (isinstance(piece, Pawn) and abs(move.final.col - move.initial.col) <= 1):
            return True
        return False

    def is_castling_move(self, move):
        piece = move.piece
        return isinstance(piece, King) and abs(move.final.col - move.initial.col) == 2

    def gives_check(self, move):
        temp_board = copy.deepcopy(self.board)
        temp_board.move(move.piece, move, testing=True)
        opponent_color = 'white' if move.piece.color == 'black' else 'black'
        return temp_board.is_king_in_check(opponent_color)

    def is_repetitive_move(self, move):
        if len(self.board.move_list) >= 4:
            last_move = self.board.move_list[-2]
            if move.initial == last_move.final and move.final == last_move.initial:
                return True
        return False

    def is_king_castled(self, king, row, col):
        castling_positions = [(7, 6), (7, 2), (0, 6), (0, 2)]
        return (row, col) in castling_positions

    def is_king_exposed(self, king, row, col):
        safety = 0
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < ROWS and 0 <= c < COLS:
                square = self.board.squares[r][c]
                if square.has_piece() and isinstance(square.piece, Pawn) and square.piece.color == king.color:
                    safety += 1
        return safety < 2

    def blocks_important_piece(self, move):
        initial_square = self.board.squares[move.initial.row][move.initial.col]
        final_square = self.board.squares[move.final.row][move.final.col]

        if isinstance(move.piece, Pawn) and (self.is_bishop_behind_pawn(initial_square) or self.is_rook_behind_pawn(initial_square)):
            return True
        return False

    def is_bishop_behind_pawn(self, square):
        row, col = square.row, square.col
        for r in range(row + 1, ROWS):
            if self.board.squares[r][col].has_piece() and isinstance(self.board.squares[r][col].piece, Bishop):
                return True
        for r in range(row - 1, -1, -1):
            if self.board.squares[r][col].has_piece() and isinstance(self.board.squares[r][col].piece, Bishop):
                return True
        return False

    def is_rook_behind_pawn(self, square):
        row, col = square.row, square.col
        for c in range(col + 1, COLS):
            if self.board.squares[row][c].has_piece() and isinstance(self.board.squares[row][c].piece, Rook):
                return True
        for c in range(col - 1, -1, -1):
            if self.board.squares[row][c].has_piece() and isinstance(self.board.squares[row][c].piece, Rook):
                return True
        return False

    def is_rook_on_open_file(self, move):
        col = move.final.col
        for r in range(ROWS):
            if self.board.squares[r][col].has_piece() and isinstance(self.board.squares[r][col].piece, Pawn):
                if self.board.squares[r][col].piece.color == move.piece.color:
                    return False
        return True

    def is_bad_pawn_move(self, move):
        if move.final.row in [0, ROWS - 1]:
            return True
        return False

    def is_central_move(self, move):
        central_squares = {(3, 3), (3, 4), (4, 3), (4, 4)}
        return (move.final.row, move.final.col) in central_squares

    def is_development_move(self, move):
        initial_row = move.initial.row
        final_row = move.final.row
        if isinstance(move.piece, (Knight, Bishop, Pawn)):
            if move.piece.color == 'white' and initial_row == 6 and final_row < 6:
                return True
            if move.piece.color == 'black' and initial_row == 1 and final_row > 1:
                return True
        return False

    def gives_check(self, move):
        temp_board = copy.deepcopy(self.board)
        temp_board.move(move.piece, move, testing=True)
        return temp_board.is_king_in_check(self.board.next_player)

    def improves_king_safety(self, move):
        return move.is_castling or (isinstance(move.piece, (Knight, Bishop, Rook)) and move.final.col in {1, 2, 6, 7})

    def creates_threat(self, move):
        temp_board = copy.deepcopy(self.board)
        temp_board.move(move.piece, move, testing=True)
        for row in range(ROWS):
            for col in range(COLS):
                if temp_board.squares[row][col].has_piece():
                    piece = temp_board.squares[row][col].piece
                    if piece.color == self.board.next_player:
                        temp_board.calc_moves(piece, row, col, bool=False)
                        for enemy_move in piece.moves:
                            if enemy_move.captured_piece:
                                return True
        return False

    def efp_pruning(self, move, depth, alpha, beta):
        if depth > 2 and self.is_less_promising(move):
            margin = (beta - alpha) / 2
            if self.ael_pruning(move, depth, alpha + margin):
                return True
        return False

    def is_less_promising(self, move):
        return move.captured_piece is not None and move.captured_piece.value < move.piece.value
