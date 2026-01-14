#!/usr/bin/env python3
"""
FluxFish Reinforcement Learning Training Pipeline
CPU-optimized self-play and training system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
import random
import os
import time
import json
from collections import deque, defaultdict
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# Import existing components
from nnue_model import FluxFishNNUE, NUM_FEATURES
from mcts import MCTS
from fast_selfplay import FastSelfPlay
from export_model import export_nnue

# ============ CPU-OPTIMIZED RL CONFIGURATION ============
@dataclass
class RLConfig:
    """CPU-optimized reinforcement learning configuration."""
    
    # Training parameters
    batch_size: int = 128  # Smaller for CPU memory
    learning_rate: float = 0.001
    epochs_per_iteration: int = 5
    max_iterations: int = 1000
    
    # Self-play parameters
    games_per_iteration: int = 20  # Games per training iteration
    max_moves_per_game: int = 200  # Prevent infinite games
    mcts_iterations: int = 500  # Reduced for CPU speed
    temperature_threshold: int = 30  # Move count before temperature drops
    
    # Experience replay
    replay_buffer_size: int = 500000  # Positions to keep
    min_buffer_size: int = 0  # Minimum positions before training (reduced from 5000)
    
    # CPU optimization
    num_workers: int = max(1, cpu_count() - 2)
    device: str = "cpu"
    
    # File paths
    model_save_path: str = "fluxfish_rl.nnue"
    experience_path: str = "rl_experience.json"
    log_path: str = "rl_training.log"
    
    # Exploration parameters
    epsilon_start: float = 0.25  # Initial exploration rate
    epsilon_end: float = 0.05   # Final exploration rate
    epsilon_decay: float = 0.999  # Decay rate per iteration

class ChessExperience:
    """Manages experience replay buffer for RL training."""
    
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.position_counts = defaultdict(int)
        
    def add_game(self, game_data: List[Tuple[str, float, List[float]]]):
        """Add a game's worth of experience."""
        for fen, value, policy in game_data:
            self.buffer.append((fen, value, policy))
            self.position_counts[fen] += 1
            
    def sample_batch(self, batch_size: int) -> List[Tuple[str, float, List[float]]]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def size(self) -> int:
        return len(self.buffer)
    
    def save(self, path: str):
        """Save experience buffer to file."""
        data = {
            'buffer': list(self.buffer),
            'position_counts': dict(self.position_counts)
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Saved {len(self.buffer)} experiences to {path}")
    
    def load(self, path: str):
        """Load experience buffer from file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.buffer = deque(data['buffer'], maxlen=self.max_size)
            self.position_counts = defaultdict(int, data['position_counts'])
            print(f"Loaded {len(self.buffer)} experiences from {path}")

class RLTrainer:
    """Main reinforcement learning trainer."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model and optimizer
        self.model = FluxFishNNUE().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        # Less aggressive scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.95, patience=20, min_lr=1e-6
        )
        
        # Fast C++ self-play
        self.fast_selfplay = FastSelfPlay()
        
        # Experience replay
        self.experience = ChessExperience(config.replay_buffer_size)
        self.experience.load(config.experience_path)
        
        # Training state
        self.iteration = 0
        self.epsilon = config.epsilon_start
        self.training_losses = []
        
        # Setup logging
        logging.basicConfig(
            filename=config.log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def board_to_tensor(self, board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Convert board to input tensors for the model."""
        # Get features from both perspectives using existing model method
        white_features = self.model.get_features(board, chess.WHITE).to(self.device)
        black_features = self.model.get_features(board, chess.BLACK).to(self.device)
        stm = board.turn == chess.WHITE
        
        return white_features, black_features, stm
    
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate position using current model."""
        with torch.no_grad():
            white_features, black_features, stm = self.board_to_tensor(board)
            value = self.model(white_features.unsqueeze(0), black_features.unsqueeze(0), stm)
            return value.item()
    
    def get_move_probabilities(self, board: chess.Board, legal_moves: List[chess.Move]) -> List[float]:
        """Get probability distribution over legal moves using MCTS with exploration."""
        # Create temporary MCTS with current model
        evaluator = lambda b: self.evaluate_position(b)
        mcts = MCTS(evaluator)
        
        # Run MCTS search
        ranked_moves = mcts.search_ranked(board, iterations=self.config.mcts_iterations)
        
        # Convert to probability distribution
        if not ranked_moves:
            return [1.0/len(legal_moves)] * len(legal_moves)
        
        # Create move -> probability mapping
        move_probs = {}
        total_visits = sum(node.n for _, node in ranked_moves)
        
        for move, node in ranked_moves:
            move_probs[move] = node.n / total_visits if total_visits > 0 else 0
        
        # Add exploration noise
        if random.random() < self.epsilon:
            for move in legal_moves:
                if move not in move_probs:
                    move_probs[move] = 0.1
            # Renormalize
            total = sum(move_probs.values())
            for move in move_probs:
                move_probs[move] /= total
        
        # Return probabilities in same order as legal_moves
        return [move_probs.get(move, 0.01) for move in legal_moves]
    
    def play_self_play_game(self) -> List[Tuple[str, float, List[float]]]:
        """Play a self-play game using fast C++ engine."""
        return self.fast_selfplay.play_fast_game(
            max_moves=self.config.max_moves_per_game,
            mcts_iterations=min(200, self.config.mcts_iterations)  # Use fewer iterations for speed
        )
    
    def train_epoch(self) -> float:
        """Train for one epoch on experience buffer."""
        if self.experience.size() < self.config.min_buffer_size:
            print(f"Skipping training - buffer size {self.experience.size()} < minimum {self.config.min_buffer_size}")
            return 0.0
        
        total_loss = 0.0
        num_batches = 0
        
        # Sample batches
        batch_size = self.config.batch_size
        num_samples = min(batch_size * 10, self.experience.size())
        batch_data = self.experience.sample_batch(num_samples)
        
        print(f"Training on {len(batch_data)} samples in batches of {batch_size}")
        
        # Create mini-batches
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i+batch_size]
            if len(batch) < 4:  # Minimum batch size
                continue
            
            # Prepare batch data
            fens = [item[0] for item in batch]
            target_values = torch.tensor([item[1] for item in batch], dtype=torch.float32).to(self.device)
            
            # Debug: Check target value range
            if num_batches == 0:  # Only print for first batch
                print(f"Target value range: [{target_values.min().item():.6f}, {target_values.max().item():.6f}]")
                print(f"Target mean: {target_values.mean().item():.6f}, std: {target_values.std().item():.6f}")
            
            # Forward pass
            predicted_values = []
            for fen in fens:
                board = chess.Board(fen)
                white_features, black_features, stm = self.board_to_tensor(board)
                value = self.model(white_features.unsqueeze(0), black_features.unsqueeze(0), stm)
                predicted_values.append(value)
            
            predicted_values = torch.cat(predicted_values).squeeze()
            
            # Ensure tensors have same shape
            if predicted_values.shape != target_values.shape:
                print(f"Shape mismatch: predicted {predicted_values.shape}, target {target_values.shape}")
                continue
            
            # Compute loss
            loss = nn.MSELoss()(predicted_values, target_values)
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss: {loss.item()}")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Stronger gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch completed: avg_loss={avg_loss:.6f}, batches={num_batches}")
        return avg_loss
    
    def train_iteration(self) -> dict:
        """Run one complete training iteration."""
        self.logger.info(f"Starting iteration {self.iteration}")
        
        # Self-play phase
        print(f"\n=== Iteration {self.iteration}: Self-Play Phase ===")
        all_game_data = []
        
        for game_num in range(self.config.games_per_iteration):
            print(f"Playing game {game_num + 1}/{self.config.games_per_iteration}...")
            game_data = self.play_self_play_game()
            all_game_data.extend(game_data)
        
        # Add to experience buffer
        self.experience.add_game(all_game_data)
        print(f"Added {len(all_game_data)} positions to experience buffer")
        print(f"Total buffer size: {self.experience.size()}")
        
        # Training phase
        print(f"\n=== Iteration {self.iteration}: Training Phase ===")
        avg_loss = 0.0
        
        for epoch in range(self.config.epochs_per_iteration):
            loss = self.train_epoch()
            avg_loss += loss
            print(f"Epoch {epoch + 1}/{self.config.epochs_per_iteration}: Loss = {loss:.6f}")
        
        avg_loss /= self.config.epochs_per_iteration
        self.training_losses.append(avg_loss)
        
        # Check for loss increase and take action
        if len(self.training_losses) >= 3:
            recent_losses = self.training_losses[-3:]
            if recent_losses[-1] > recent_losses[0] * 1.5:  # 50% increase
                print("⚠️  WARNING: Loss increased significantly!")
                print(f"   Recent losses: {[f'{l:.6f}' for l in recent_losses]}")
                
                # Emergency: Reduce learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * 0.5, 1e-6)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"   Emergency LR reduction: {current_lr:.6f} → {new_lr:.6f}")
        
        self.scheduler.step(avg_loss)
        
        # Decay exploration rate
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        
        # Save checkpoint
        if self.iteration % 5 == 0:
            self.save_checkpoint()
        
        # Log progress
        stats = {
            'iteration': self.iteration,
            'loss': avg_loss,
            'epsilon': self.epsilon,
            'buffer_size': self.experience.size(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        self.logger.info(f"Iteration {self.iteration} completed: {stats}")
        print(f"\nIteration {self.iteration} completed:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Exploration Rate: {self.epsilon:.3f}")
        print(f"  Buffer Size: {self.experience.size()}")
        print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        self.iteration += 1
        return stats
    
    def save_checkpoint(self):
        """Save model and training state."""
        # Save PyTorch model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': self.iteration,
            'epsilon': self.epsilon,
            'training_losses': self.training_losses
        }, self.config.model_save_path)
        
        # Export to .bin for C++ backend
        bin_path = "fluxfish.bin"
        export_nnue(self.config.model_save_path, bin_path)
        
        # Save experience
        self.experience.save(self.config.experience_path)
        
        print(f"Checkpoint saved at iteration {self.iteration}")
        print(f"  PyTorch model: {self.config.model_save_path}")
        print(f"  C++ binary: {bin_path}")
        print(f"  Experience: {self.config.experience_path}")
    
    def load_checkpoint(self):
        """Load model and training state."""
        if os.path.exists(self.config.model_save_path):
            checkpoint = torch.load(self.config.model_save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.iteration = checkpoint.get('iteration', 0)
            self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
            self.training_losses = checkpoint.get('training_losses', [])
            print(f"Loaded checkpoint from iteration {self.iteration}")
    
    def train(self):
        """Main training loop."""
        print("=== FluxFish Reinforcement Learning Training ===")
        print(f"Device: {self.device}")
        print(f"CPU Cores: {self.config.num_workers}")
        print(f"Buffer Size: {self.config.replay_buffer_size}")
        print(f"Games per Iteration: {self.config.games_per_iteration}")
        
        # Load existing checkpoint
        self.load_checkpoint()
        
        try:
            while self.iteration < self.config.max_iterations:
                stats = self.train_iteration()
                
                # Early stopping if loss is very low
                if len(self.training_losses) >= 10:
                    recent_losses = self.training_losses[-10:]
                    if all(loss < 0.001 for loss in recent_losses):
                        print("Training converged! Stopping early.")
                        break
                        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self.save_checkpoint()
        
        print("\nTraining completed!")
        self.save_checkpoint()

def main():
    """Main entry point for RL training."""
    config = RLConfig()
    
    # CPU-specific optimizations
    print("Applying CPU optimizations...")
    torch.set_num_threads(config.num_workers)
    
    trainer = RLTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
