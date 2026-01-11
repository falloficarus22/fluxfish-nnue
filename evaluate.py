import numpy as np
import chess
from loader import NNUELoader

class Evaluator:
    def __init__(self, model_path):
        # SF 15.1 "Big" architecture
        self.HIDDEN_SIZE = 512 # Most likely for this size
        self.INPUT_SIZE = 49152 # 64 king sq * 768 pieces
        
        loader = NNUELoader(model_path)
        # Try to load. If it fails due to size, we will catch it.
        try:
            self.weights = loader.load_full_sf15(self.INPUT_SIZE, self.HIDDEN_SIZE)
        except Exception as e:
            print(f"Failed with H=512, trying H=768... ({e})")
            loader = NNUELoader(model_path)
            self.HIDDEN_SIZE = 768
            self.weights = loader.load_full_sf15(self.INPUT_SIZE, self.HIDDEN_SIZE)

        print(f"Successfully loaded NNUE with Hidden Size: {self.HIDDEN_SIZE}")

    def get_indices(self, board, color):
        """Standard HalfKAv2 indexing"""
        indices = []
        king_sq = board.king(color)
        
        # Mirror king square for black
        k_sq = king_sq if color == chess.WHITE else chess.square_mirror(king_sq)
        
        for sq, piece in board.piece_map().items():
            if piece.piece_type == chess.KING: continue
            
            # 1. Perspective
            p_sq = sq if color == chess.WHITE else chess.square_mirror(sq)
            p_color = piece.color if color == chess.WHITE else not piece.color
            
            # 2. Piece mapping (Pawn=0, Knight=1, ..., Queen=4)
            # SF uses: 0:P, 1:N, 2:B, 3:R, 4:Q. King is excluded.
            # Mirror pieces if black's perspective
            p_idx = piece.piece_type - 1 # 0 to 4
            
            color_offset = 0 if p_color == chess.WHITE else 384
            piece_offset = p_idx * 64
            
            idx = (k_sq * 768) + color_offset + piece_offset + p_sq
            indices.append(idx)
        return indices

    def evaluate(self, board):
        # 1. Accumulators
        w_idx = self.get_indices(board, chess.WHITE)
        b_idx = self.get_indices(board, chess.BLACK)
        
        w_acc = self.weights['ft_b'].copy()
        for i in w_idx: w_acc += self.weights['ft_w'][i]
        
        b_acc = self.weights['ft_b'].copy()
        for i in b_idx: b_acc += self.weights['ft_w'][i]
        
        # 2. Activation (ClippedReLU 0-127)
        w_hidden = np.clip(w_acc, 0, 127)
        b_hidden = np.clip(b_acc, 0, 127)
        
        # 3. Concatenate
        if board.turn == chess.WHITE:
            x = np.concatenate([w_hidden, b_hidden])
        else:
            x = np.concatenate([b_hidden, w_hidden])
            
        # 4. Head Layers (with fixed-point shifts)
        # Layer 1
        x = np.dot(self.weights['l1_w'], x) // 64 + self.weights['l1_b']
        x = np.clip(x, 0, 127)
        
        # Layer 2
        x = np.dot(self.weights['l2_w'], x) // 64 + self.weights['l2_b']
        x = np.clip(x, 0, 127)
        
        # Layer 3
        v = np.dot(self.weights['l3_w'], x) // 64 + self.weights['l3_b']
        
        # Scale to centipawns (roughly) then to tanh
        # SF evaluation is scaled such that 1.0 = ~300cp in MCTS
        return np.tanh(v[0] / 300.0)