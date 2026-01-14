#!/usr/bin/env python3
"""
Simple RL training without PyTorch dependency issues.
Uses numpy-based neural network implementation.
"""

import numpy as np
import chess
import json
import random
import os
from collections import deque, defaultdict
from typing import List, Tuple
from fast_selfplay import FastSelfPlay

# Simple neural network using numpy
class SimpleNNUE:
    """Simple NNUE implementation using numpy."""
    
    def __init__(self, input_size=1536, hidden_size=256, output_size=1):
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size * 2, hidden_size) * 0.01
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size) * 0.01
        self.b3 = np.zeros(output_size)
        
        self.learning_rate = 0.001
        
    def get_features(self, board: chess.Board, perspective: chess.Color) -> np.ndarray:
        """Extract 768-dim features."""
        features = np.zeros(768)
        
        for sq, piece in board.piece_map().items():
            square = sq if perspective == chess.WHITE else chess.square_mirror(sq)
            rel_color = 0 if piece.color == perspective else 1
            idx = (piece.piece_type - 1) * 128 + rel_color * 64 + square
            if 0 <= idx < 768:
                features[idx] = 1.0
                
        return features
    
    def forward(self, white_features: np.ndarray, black_features: np.ndarray, stm: bool) -> float:
        """Forward pass."""
        # Feature transformer
        h1_white = np.clip(white_features @ self.W1 + self.b1, 0, 1)
        h1_black = np.clip(black_features @ self.W1 + self.b1, 0, 1)
        
        # Concatenate based on side to move
        if stm:
            h_concat = np.concatenate([h1_white, h1_black])
        else:
            h_concat = np.concatenate([h1_black, h1_white])
        
        # Hidden layers
        h2 = np.clip(h_concat @ self.W2 + self.b2, 0, 1)
        output = np.tanh(h2 @ self.W3 + self.b3)
        
        return float(output[0])
    
    def train_step(self, white_features: np.ndarray, black_features: np.ndarray, 
                  stm: bool, target_value: float):
        """Single training step using simple gradient descent."""
        # Forward pass
        h1_white = np.clip(white_features @ self.W1 + self.b1, 0, 1)
        h1_black = np.clip(black_features @ self.W1 + self.b1, 0, 1)
        
        if stm:
            h_concat = np.concatenate([h1_white, h1_black])
        else:
            h_concat = np.concatenate([h1_black, h1_white])
        
        h2 = np.clip(h_concat @ self.W2 + self.b2, 0, 1)
        output = np.tanh(h2 @ self.W3 + self.b3)
        
        # Compute loss (MSE)
        loss = 0.5 * (output[0] - target_value) ** 2
        
        # Backward pass (simplified gradients)
        output_error = output[0] - target_value
        
        # Gradient for output layer
        dW3 = np.outer(h2, output_error * (1 - output[0]**2))
        db3 = output_error * (1 - output[0]**2)
        
        # Gradient for hidden layer 2
        h2_error = (output_error * (1 - output[0]**2)) @ self.W3.T
        h2_error = h2_error * (h2 > 0)  # ReLU derivative
        
        dW2 = np.outer(h_concat, h2_error)
        db2 = h2_error
        
        # Gradient for hidden layer 1 (simplified)
        h1_error = h2_error @ self.W2.T[:768]  # Only take first part
        h1_error = h1_error * (h1_white > 0)
        
        dW1 = np.outer(white_features, h1_error) + np.outer(black_features, h1_error)
        db1 = h1_error
        
        # Update weights
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
        return float(loss)

class SimpleRLTrainer:
    """Simple RL trainer using numpy."""
    
    def __init__(self):
        self.model = SimpleNNUE()
        self.experience = deque(maxlen=10000)
        self.fast_selfplay = FastSelfPlay()
        self.iteration = 0
        
    def collect_games(self, num_games: int):
        """Collect self-play games."""
        print(f"Collecting {num_games} games...")
        
        for game_num in range(num_games):
            print(f"Game {game_num + 1}/{num_games}")
            game_data = self.fast_selfplay.play_fast_game(max_moves=50, mcts_iterations=100)
            
            for fen, value, policy in game_data:
                self.experience.append((fen, value))
        
        print(f"Collected {len(self.experience)} positions")
    
    def train_epoch(self, batch_size: int = 32, num_batches: int = 100):
        """Train for one epoch."""
        if len(self.experience) < batch_size:
            return 0.0
        
        total_loss = 0.0
        
        for batch_num in range(num_batches):
            # Sample batch
            batch = random.sample(list(self.experience), batch_size)
            
            batch_loss = 0.0
            for fen, target_value in batch:
                board = chess.Board(fen)
                
                # Get features
                white_features = self.model.get_features(board, chess.WHITE)
                black_features = self.model.get_features(board, chess.BLACK)
                stm = board.turn == chess.WHITE
                
                # Train step
                loss = self.model.train_step(white_features, black_features, stm, target_value)
                batch_loss += loss
            
            total_loss += batch_loss / batch_size
            
            if batch_num % 20 == 0:
                print(f"  Batch {batch_num}/{num_batches}, Loss: {batch_loss/batch_size:.6f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch completed, Average Loss: {avg_loss:.6f}")
        return avg_loss
    
    def train_iteration(self):
        """One training iteration."""
        print(f"\n=== Iteration {self.iteration} ===")
        
        # Collect games
        self.collect_games(num_games=5)
        
        # Train
        avg_loss = self.train_epoch(batch_size=16, num_batches=50)
        
        self.iteration += 1
        return avg_loss
    
    def save_model(self, path: str):
        """Save model weights."""
        weights = {
            'W1': self.model.W1.tolist(),
            'b1': self.model.b1.tolist(),
            'W2': self.model.W2.tolist(),
            'b2': self.model.b2.tolist(),
            'W3': self.model.W3.tolist(),
            'b3': self.model.b3.tolist(),
            'iteration': self.iteration
        }
        
        with open(path, 'w') as f:
            json.dump(weights, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                weights = json.load(f)
            
            self.model.W1 = np.array(weights['W1'])
            self.model.b1 = np.array(weights['b1'])
            self.model.W2 = np.array(weights['W2'])
            self.model.b2 = np.array(weights['b2'])
            self.model.W3 = np.array(weights['W3'])
            self.model.b3 = np.array(weights['b3'])
            self.iteration = weights.get('iteration', 0)
            print(f"Model loaded from {path}, iteration {self.iteration}")

def main():
    """Main training loop."""
    print("=== Simple RL Training (No PyTorch) ===")
    
    trainer = SimpleRLTrainer()
    trainer.load_model("simple_rl_model.json")
    
    try:
        for i in range(10):  # 10 iterations
            loss = trainer.train_iteration()
            
            # Save every 2 iterations
            if i % 2 == 0:
                trainer.save_model("simple_rl_model.json")
            
            print(f"Iteration {i+1} completed, Loss: {loss:.6f}")
            
            # Stop if loss is very low
            if loss < 0.01:
                print("Training converged!")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        trainer.save_model("simple_rl_model.json")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
