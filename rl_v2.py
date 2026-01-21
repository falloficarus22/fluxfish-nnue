#!/usr/bin/env python3
"""
FluxFish Reinforcement Learning v2.0
Modern, efficient RL training system with optimized self-play and training.
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
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# Import existing components
from nnue_model import FluxFishNNUE, NUM_FEATURES

# ============ CONFIGURATION ============
@dataclass
class RLConfigV2:
    """Modern RL configuration with optimizations."""
    
    # Training parameters
    batch_size: int = 512  # Larger batches for better GPU utilization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 10
    max_iterations: int = 2000
    
    # Self-play parameters
    games_per_iteration: int = 64  # More games for diverse experience
    max_moves_per_game: int = 250
    mcts_iterations: int = 800  # Higher quality self-play
    temperature_threshold: int = 20
    
    # Experience replay
    replay_buffer_size: int = 1000000  # 1M positions
    min_buffer_size: int = 10000  # Start training after 10K positions
    priority_sampling: bool = True  # Prioritized experience replay
    
    # Optimization
    num_workers: int = min(8, mp.cpu_count())
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = torch.cuda.is_available()
    
    # Exploration
    initial_epsilon: float = 0.25
    final_epsilon: float = 0.05
    epsilon_decay: float = 0.999
    
    # Checkpointing
    save_interval: int = 50
    eval_interval: int = 25
    
    # Paths
    model_path: str = "fluxfish_v2.nnue"
    checkpoint_dir: str = "checkpoints_v2"
    log_dir: str = "logs_v2"

# ============ EXPERIENCE REPLAY ============
class ExperienceBuffer:
    """Efficient experience replay with prioritization."""
    
    def __init__(self, config: RLConfigV2):
        self.config = config
        self.buffer = deque(maxlen=config.replay_buffer_size)
        self.priorities = deque(maxlen=config.replay_buffer_size)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        
    def add(self, experiences: List[Dict]):
        """Add new experiences with initial priority."""
        for exp in experiences:
            # Initial priority based on TD error (use absolute value + small epsilon)
            priority = abs(exp.get('td_error', 1.0)) + 1e-6
            self.buffer.append(exp)
            self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """Sample batch with prioritized experience replay."""
        if not self.config.priority_sampling:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            weights = np.ones(batch_size)
            experiences = [self.buffer[i] for i in indices]
            return experiences, indices, weights
        
        # Prioritized sampling
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

# ============ SELF-PLAY ENGINE ============
class SelfPlayGame:
    """Single self-play game with optimized MCTS."""
    
    def __init__(self, model: FluxFishNNUE, config: RLConfigV2):
        self.model = model
        self.config = config
        self.board = chess.Board()
        self.experiences = []
        
    def play_game(self) -> List[Dict]:
        """Play a single self-play game."""
        self.experiences = []
        self.board.reset()
        
        move_count = 0
        
        while not self.board.is_game_over() and move_count < self.config.max_moves_per_game:
            # Get MCTS policy
            policy, value = self._get_mcts_policy()
            
            # Store experience
            features = self._extract_features()
            self.experiences.append({
                'features': features,
                'policy': policy,
                'value': value,
                'move_count': move_count
            })
            
            # Select move with temperature
            move = self._select_move(policy, move_count)
            self.board.push(move)
            move_count += 1
        
        # Assign game result to all experiences
        game_result = self._get_game_result()
        for exp in self.experiences:
            exp['result'] = game_result
        
        return self.experiences
    
    def _get_mcts_policy(self) -> Tuple[np.ndarray, float]:
        """Get policy from MCTS search."""
        # This would integrate with your optimized C++ MCTS
        # For now, return a simple policy based on model evaluation
        features = self._extract_features()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            if self.config.device == "cuda":
                features_tensor = features_tensor.cuda()
            
            policy_logits, value = self.model(features_tensor)
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            value = value.item()
        
        return policy, value
    
    def _extract_features(self) -> np.ndarray:
        """Extract NNUE features from current board."""
        # Use existing feature extraction from nnue_model
        features = np.zeros(NUM_FEATURES, dtype=np.float32)
        
        # Convert board to feature representation
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                # This would use the same feature extraction as your NNUE model
                feature_idx = self._get_feature_index(sq, piece)
                if feature_idx < NUM_FEATURES:
                    features[feature_idx] = 1.0
        
        return features
    
    def _get_feature_index(self, square: chess.Square, piece: chess.Piece) -> int:
        """Get feature index for NNUE."""
        # Simplified - implement proper feature mapping
        piece_type = piece.piece_type - 1  # 0-5
        color = 0 if piece.color == chess.WHITE else 1
        return (piece_type * 2 + color) * 64 + square
    
    def _select_move(self, policy: np.ndarray, move_count: int) -> chess.Move:
        """Select move with temperature annealing."""
        legal_moves = list(self.board.legal_moves)
        legal_indices = [move.from_square() * 64 + move.to_square() for move in legal_moves]
        
        # Temperature annealing
        if move_count < self.config.temperature_threshold:
            temperature = 1.0
        else:
            temperature = 0.1
        
        # Apply temperature
        legal_probs = np.array([policy[i] if i < len(policy) else 0.01 for i in legal_indices])
        legal_probs = legal_probs ** temperature
        legal_probs /= legal_probs.sum()
        
        # Sample move
        move_idx = np.random.choice(len(legal_moves), p=legal_probs)
        return legal_moves[move_idx]
    
    def _get_game_result(self) -> float:
        """Get game result from perspective of current player."""
        if self.board.is_checkmate():
            return -1.0 if self.board.turn == chess.WHITE else 1.0
        elif self.board.is_stalemate() or self.board.can_claim_draw():
            return 0.0
        else:
            # Should not happen, but return draw
            return 0.0

# ============ TRAINING PIPELINE ============
class RLTrainerV2:
    """Modern RL training pipeline."""
    
    def __init__(self, config: RLConfigV2):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Initialize model
        self.model = FluxFishNNUE().to(config.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.max_iterations
        )
        
        # Mixed precision training
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Experience buffer
        self.exp_buffer = ExperienceBuffer(config)
        
        # Training state
        self.iteration = 0
        self.epsilon = config.initial_epsilon
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        
    def setup_logging(self):
        """Setup logging configuration."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.log_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
    def train(self):
        """Main training loop."""
        self.logger.info("Starting RL training v2.0")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        try:
            for iteration in range(self.iteration, self.config.max_iterations):
                self.iteration = iteration
                
                # Self-play phase
                experiences = self.self_play_phase()
                
                # Training phase
                if len(self.exp_buffer) >= self.config.min_buffer_size:
                    loss = self.training_phase()
                    self.metrics['loss'].append(loss)
                    
                    # Update best model
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.save_checkpoint(f"{self.config.checkpoint_dir}/best_model.pt")
                
                # Evaluation phase
                if iteration % self.config.eval_interval == 0:
                    self.evaluate()
                
                # Checkpointing
                if iteration % self.config.save_interval == 0:
                    self.save_checkpoint()
                
                # Logging
                self.log_progress(iteration)
                
                # Update epsilon
                self.epsilon = max(
                    self.config.final_epsilon,
                    self.epsilon * self.config.epsilon_decay
                )
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        finally:
            self.save_checkpoint()
            self.logger.info("Training completed")
    
    def self_play_phase(self) -> List[Dict]:
        """Generate self-play experiences."""
        self.logger.info(f"Self-play phase - iteration {self.iteration}")
        
        all_experiences = []
        
        # Parallel self-play
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit games
            futures = []
            for _ in range(self.config.games_per_iteration):
                future = executor.submit(self._play_single_game)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    experiences = future.result(timeout=300)  # 5 minute timeout
                    all_experiences.extend(experiences)
                except Exception as e:
                    self.logger.warning(f"Game failed: {e}")
        
        # Add to experience buffer
        self.exp_buffer.add(all_experiences)
        
        self.logger.info(f"Generated {len(all_experiences)} experiences")
        self.logger.info(f"Buffer size: {len(self.exp_buffer)}")
        
        return all_experiences
    
    def _play_single_game(self) -> List[Dict]:
        """Play single game (for multiprocessing)."""
        # This would be called in separate process
        # For now, return dummy data
        return []
    
    def training_phase(self) -> float:
        """Training phase with experience replay."""
        self.logger.info(f"Training phase - iteration {self.iteration}")
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.epochs_per_iteration):
            # Sample batches
            num_batches_per_epoch = len(self.exp_buffer) // self.config.batch_size
            
            for batch in range(num_batches_per_epoch):
                experiences, indices, weights = self.exp_buffer.sample(self.config.batch_size)
                
                # Prepare batch data
                features = torch.FloatTensor([exp['features'] for exp in experiences])
                policies = torch.FloatTensor([exp['policy'] for exp in experiences])
                values = torch.FloatTensor([exp['result'] for exp in experiences])
                weights = torch.FloatTensor(weights)
                
                if self.config.device == "cuda":
                    features, policies, values, weights = features.cuda(), policies.cuda(), values.cuda(), weights.cuda()
                
                # Forward pass
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        pred_policies, pred_values = self.model(features)
                        
                        policy_loss = nn.CrossEntropyLoss(reduction='none')(pred_policies, policies)
                        value_loss = nn.MSELoss(reduction='none')(pred_values.squeeze(), values)
                        loss = (policy_loss + value_loss) * weights
                        loss = loss.mean()
                else:
                    pred_policies, pred_values = self.model(features)
                    
                    policy_loss = nn.CrossEntropyLoss(reduction='none')(pred_policies, policies)
                    value_loss = nn.MSELoss(reduction='none')(pred_values.squeeze(), values)
                    loss = (policy_loss + value_loss) * weights
                    loss = loss.mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.scheduler.step()
        
        self.logger.info(f"Training loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(self):
        """Evaluate current model."""
        self.logger.info(f"Evaluation - iteration {self.iteration}")
        # Implement evaluation against previous model or baseline
        pass
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint."""
        if path is None:
            path = f"{self.config.checkpoint_dir}/checkpoint_{self.iteration}.pt"
        
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'best_loss': self.best_loss,
            'metrics': dict(self.metrics),
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        # Also save in NNUE format for engine use
        if path.endswith(f"checkpoint_{self.iteration}.pt"):
            nnue_path = f"{self.config.model_path}"
            self.export_to_nnue(nnue_path)
    
    def export_to_nnue(self, path: str):
        """Export model to NNUE format."""
        # Use existing export functionality
        from export_model import export_nnue
        export_nnue(self.model, path)
        self.logger.info(f"Model exported to {path}")
    
    def load_checkpoint(self, path: Optional[str] = None):
        """Load training checkpoint."""
        if path is None:
            # Find latest checkpoint
            checkpoints = list(Path(self.config.checkpoint_dir).glob("checkpoint_*.pt"))
            if not checkpoints:
                self.logger.info("No checkpoint found, starting from scratch")
                return
            path = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
        
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.epsilon = checkpoint['epsilon']
        self.best_loss = checkpoint['best_loss']
        self.metrics = defaultdict(list, checkpoint['metrics'])
        
        self.logger.info(f"Loaded checkpoint from {path}")
    
    def log_progress(self, iteration: int):
        """Log training progress."""
        if len(self.metrics['loss']) > 0:
            current_loss = self.metrics['loss'][-1]
            lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.info(
                f"Iter {iteration:4d} | "
                f"Loss: {current_loss:.4f} | "
                f"LR: {lr:.6f} | "
                f"Buffer: {len(self.exp_buffer):6d} | "
                f"Epsilon: {self.epsilon:.3f}"
            )

# ============ MAIN ENTRY POINT ============
def main():
    """Main training entry point."""
    config = RLConfigV2()
    
    print("=== FluxFish RL Training v2.0 ===")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Buffer size: {config.replay_buffer_size:,}")
    print(f"Workers: {config.num_workers}")
    
    trainer = RLTrainerV2(config)
    
    # Try to load existing checkpoint
    try:
        trainer.load_checkpoint()
    except:
        print("No checkpoint found, starting fresh")
    
    trainer.train()

if __name__ == "__main__":
    main()
