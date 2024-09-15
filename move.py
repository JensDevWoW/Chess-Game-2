class Move:
    def __init__(self, initial, final, piece, captured_piece=None, promotion=None, is_en_passant=False, is_castling=False):
        self.initial = initial  # Starting square of the move
        self.final = final      # Ending square of the move
        self.piece = piece      # The piece being moved
        self.captured_piece = captured_piece  # Any piece captured by this move
        self.promotion = promotion  # If this move is a promotion, set the promoted piece type
        self.is_en_passant = is_en_passant  # Flag if this move is an en passant capture
        self.is_castling = is_castling  # Flag if this move is castling
        self.weight = 0  # For engine: Heuristic or evaluation score for the move

    def __str__(self):
        s = f'({self.initial.col}, {self.initial.row}) -> ({self.final.col}, {self.final.row})'
        if self.promotion:
            s += f' promoting to {self.promotion}'
        if self.is_en_passant:
            s += ' (en passant)'
        if self.is_castling:
            s += ' (castling)'
        return s

    def __eq__(self, other):
        return (
            self.initial == other.initial and
            self.final == other.final and
            self.promotion == other.promotion and
            self.is_en_passant == other.is_en_passant and
            self.is_castling == other.is_castling
        )
