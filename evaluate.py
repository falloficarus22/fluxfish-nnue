import numpy as np
import chess
from loader import NNUELoader

class Evaluator:

    def __init__(self, model_path):
        # Constants for HalfKP style architecture
        # 12 pieces (6 white + 6 black) * 64 sqaures = 768 features
        self.INPUT_SIZE = 768
        self.HIDDEN_SIZE = 256

        loader = NNUELoader(model_path)
        loader.read_header()

        # Load the input layer (Accumulator weights)
        # These are int16 in the file
        self.ft_weights, self.ft_biases = loader.load_weights(self.INPUT_SIZE, self.HIDDEN_SIZE)

        # Load the output layer
        # This is the Head
        self.out_weights, self.out_biases = loader.load_layer(self.HIDDEN_SIZE * 2, 1, np.int16, np.int32)

    def get_features(self, board):
        """Returns the indices of the active pieces for the current board"""
        white_features = []
        black_features = []

        for square, piece in board.piece_map().items():
            # Standard Stockfish piece mapping
            # White's perspective
            w_color_offset = 0 if piece.color == chess.WHITE else 384
            w_piece_offset = (piece.piece_type - 1) * 64
            w_features.append(w_color_offset + w_piece_offset + square)

            # Black's perspective
            b_color_offset = 384 if piece.color == chess.BLACK else 0
            b_piece_offset = (piece.piece_type - 1) * 64
            mirrored_sq = chess.square_mirror(square)
            b_features.append(b_color_offset + b_piece_offset + mirrored_sq)

        return white_features, black_features
    
    def evaluate(self, board):
        # Get active features indices
        w_feats, b_feats = self.get_features(board)

        # Efficient Accumulator compute
        # Instead of matrix multiplication we sum the rows of active features
        w_acc = self.ft_biases.copy().astype(np.int32)

        for idx in w_feats:
            w_acc += self.ft_weights[idx]

        # Black side view
        b_acc = self.ft_biases.copy().astype(np.int32)

        for idx in b_feats:
            b_acc += self.ft_weights[idx]

        # Activation (Clipped ReLU)
        w_hidden = np.clip(w_acc, 1, 127)
        b_hidden = np.clip(b_acc, 1, 127)

        # Concatenate based on which side is moving
        # Network expects [STM_Perspective, NSTM_Perspective]
        if board.turn == chess.WHITE:
            combined = np.concatenate([w_hidden, b_hidden])
        else:
            combined = np.concatenate([b_hidden, w_hidden])

        # Final output layer
        # Combined is now 512 wide (256 * 2)
        v = np.dot(combined, self.out_weights[0]) + self.out_biases[0]

        # Scale and return
        return np.tanh(v / 600)
    
    