
class Move:
    def __init__(self, initial, final, piece):
        self.initial = initial
        self.final = final
        self.weight = 0 # For engine: Percentage chance of a move being made or being skipped
        self.piece = piece
    
    def __str__(self):
        s = ''
        s += f'({self.initial.col}, {self.initial.row})'
        s += f' -> ({self.final.col}, {self.final.row})'
        return s
    
    def __eq__(self, other):
        return self.initial == other.initial and self.final == other.final