# zobrist.py
import random

# Define constants for pieces and colors
PIECES = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
COLORS = ['white', 'black']

# Create a dictionary to store Zobrist keys
ZOBRIST_KEYS = {}

def initialize_zobrist_keys():
    """Initializes Zobrist keys for each piece type, color, and square on the board."""
    for color in COLORS:
        ZOBRIST_KEYS[color] = {}
        for piece in PIECES:
            # Create a 2D list (8x8) of random 64-bit integers for each piece type and color
            ZOBRIST_KEYS[color][piece] = [[random.getrandbits(64) for _ in range(8)] for _ in range(8)]

# Call the function to initialize keys at module load
initialize_zobrist_keys()
