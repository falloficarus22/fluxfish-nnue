"""
FluxFish NNUE - Lightweight trainable neural network for chess evaluation.
Uses a simplified architecture for fast training.
"""

import torch
import torch.nn as nn
import numpy as np
import chess

# Feature encoding: 768 features (6 piece types * 2 colors * 64 squares)
NUM_FEATURES = 768
HIDDEN_SIZE = 256

class FluxFishNNUE(nn.Module):
    """
    A simplified NNUE architecture optimized for fast training:
    - Input: 768-dim piece-square features (no king conditioning for speed)
    - Architecture: 768 -> 256 -> 32 -> 1
    - ClippedReLU activations
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature transformer (accumulator layer)
        self.ft = nn.Linear(NUM_FEATURES, HIDDEN_SIZE)
        
        # Output head
        self.l1 = nn.Linear(HIDDEN_SIZE * 2, 32)  # *2 for both perspectives
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)
        
        # Initialize weights for better initial behavior
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_features(self, board: chess.Board, perspective: chess.Color) -> torch.Tensor:
        """
        Extract piece-square features from the board.
        
        Feature index = piece_type * 128 + color * 64 + square
        - piece_type: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
        - color: 0=White, 1=Black (from perspective's view)
        - square: 0-63 (mirrored for black's perspective)
        """
        features = torch.zeros(NUM_FEATURES)
        
        for sq, piece in board.piece_map().items():
            # Mirror square for black's perspective
            square = sq if perspective == chess.WHITE else chess.square_mirror(sq)
            
            # Relative color (0 = friendly, 1 = enemy)
            rel_color = 0 if piece.color == perspective else 1
            
            # Feature index
            idx = (piece.piece_type - 1) * 128 + rel_color * 64 + square
            features[idx] = 1.0
            
        return features
    
    def forward_accumulator(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature transformer."""
        x = self.ft(features)
        x = torch.clamp(x, 0, 1)  # ClippedReLU
        return x
    
    def forward(self, white_features: torch.Tensor, black_features: torch.Tensor, 
                stm: bool) -> torch.Tensor:
        """
        Full forward pass.
        
        Args:
            white_features: Features from white's perspective
            black_features: Features from black's perspective  
            stm: True if white to move, False if black to move
        """
        w_acc = self.forward_accumulator(white_features)
        b_acc = self.forward_accumulator(black_features)
        
        # Concatenate based on side to move
        if stm:  # White to move
            x = torch.cat([w_acc, b_acc], dim=-1)
        else:  # Black to move
            x = torch.cat([b_acc, w_acc], dim=-1)
        
        # Output head
        x = torch.clamp(self.l1(x), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = torch.tanh(self.l3(x))  # Output in [-1, 1]
        
        return x


class FluxFishEvaluator:
    """Wrapper class for using the NNUE model for evaluation."""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = FluxFishNNUE().to(self.device)
        
        if model_path:
            self.load(model_path)
        else:
            print("No model loaded - using random weights!")
        
        self.model.eval()
    
    def load(self, path: str):
        """Load model weights from file."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded NNUE model from {path}")
    
    def save(self, path: str):
        """Save model weights to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)
        print(f"Saved NNUE model to {path}")
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a chess position.
        
        Returns:
            float: Evaluation in [-1, 1] from current side's perspective
        """
        with torch.no_grad():
            w_feat = self.model.get_features(board, chess.WHITE).to(self.device)
            b_feat = self.model.get_features(board, chess.BLACK).to(self.device)
            
            value = self.model(w_feat, b_feat, board.turn == chess.WHITE)
            return value.item()


def batch_get_features(boards: list, device='cpu') -> tuple:
    """
    Batch feature extraction for efficient training.
    
    Returns:
        Tuple of (white_features, black_features, stm, labels) tensors
    """
    n = len(boards)
    w_features = torch.zeros(n, NUM_FEATURES)
    b_features = torch.zeros(n, NUM_FEATURES)
    stm = torch.zeros(n, dtype=torch.bool)
    
    model = FluxFishNNUE()  # Just for feature extraction
    
    for i, board in enumerate(boards):
        w_features[i] = model.get_features(board, chess.WHITE)
        b_features[i] = model.get_features(board, chess.BLACK)
        stm[i] = board.turn == chess.WHITE
    
    return w_features.to(device), b_features.to(device), stm.to(device)
