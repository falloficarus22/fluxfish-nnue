#!/usr/bin/env python3
"""
FluxFish RL Training - Final Version
Uses C++ backend (fluxfish.bin) for self-play, trains new model as fluxfish.nnue
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import existing components
from nnue_model import FluxFishNNUE, NUM_FEATURES
from export_model import export_nnue
from cpp_selfplay import CppSelfPlay

@dataclass
class RLConfig:
    """Configuration for C++ backend RL training."""
    
    # Training parameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 3
    max_iterations: int = 100
    
    # Self-play parameters
    games_per_iteration: int = 8
    max_moves_per_game: int = 200
    time_per_move_ms: int = 50
    
    # Experience replay
    replay_buffer_size: int = 50000
    min_buffer_size: int = 1000
    
    # Optimization
    num_workers: int = min(4, mp.cpu_count())
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    cpp_engine_path: str = "./cpp/fluxfish_cpp"
    model_output_path: str = "fluxfish.nnue"
    checkpoint_dir: str = "checkpoints_rl"
    log_dir: str = "logs_rl"

class ExperienceBuffer:
    """Simple experience replay buffer."""
    
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, experiences: List[Dict]):
        for exp in experiences:
            self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> List[Dict]:
        if len(self.buffer) < batch_size:
            return []
        return np.random.choice(list(self.buffer), batch_size, replace=False).tolist()
    
    def __len__(self):
        return len(self.buffer)

class RLTrainer:
    """Main RL trainer using C++ backend."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Initialize new model for training
        self.model = FluxFishNNUE().to(config.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Experience buffer
        self.exp_buffer = ExperienceBuffer(config.replay_buffer_size)
        
        # Self-play engine
        self.selfplay = CppSelfPlay(config.cpp_engine_path)
        
        # Training state
        self.iteration = 0
        self.best_loss = float('inf')
        
    def setup_logging(self):
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
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
    def train(self):
        """Main training loop."""
        self.logger.info("=== Starting FluxFish RL Training ===")
        self.logger.info(f"Using C++ engine: {self.config.cpp_engine_path}")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Output model: {self.config.model_output_path}")
        
        try:
            for iteration in range(self.iteration, self.config.max_iterations):
                self.iteration = iteration
                
                # Phase 1: Self-play with C++ backend
                experiences = self.self_play_phase()
                
                # Phase 2: Training on collected experiences
                if len(self.exp_buffer) >= self.config.min_buffer_size:
                    loss = self.training_phase()
                    
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.save_model()
                        self.logger.info(f"New best loss: {loss:.4f}")
                
                self.log_progress()
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        finally:
            self.save_model()
            self.logger.info("Training completed")
    
    def self_play_phase(self) -> List[Dict]:
        """Generate self-play games using C++ engine."""
        self.logger.info(f"Self-play iteration {self.iteration}")
        
        all_experiences = []
        
        # Parallel self-play
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for _ in range(self.config.games_per_iteration):
                future = executor.submit(
                    self.selfplay.play_game,
                    self.config.max_moves_per_game,
                    self.config.time_per_move_ms
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    experiences = future.result(timeout=120)  # 2 minute timeout
                    all_experiences.extend(experiences)
                except Exception as e:
                    self.logger.warning(f"Game failed: {e}")
        
        # Add to buffer
        self.exp_buffer.add(all_experiences)
        
        self.logger.info(f"Generated {len(all_experiences)} experiences")
        self.logger.info(f"Buffer size: {len(self.exp_buffer)}")
        
        return all_experiences
    
    def training_phase(self) -> float:
        """Train neural network on collected experiences."""
        self.logger.info("Training phase")
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.epochs_per_iteration):
            # Calculate number of batches
            num_batches_per_epoch = len(self.exp_buffer) // self.config.batch_size
            
            for batch_idx in range(num_batches_per_epoch):
                # Sample batch
                batch = self.exp_buffer.sample(self.config.batch_size)
                if not batch:
                    continue
                
                # Prepare batch data
                features = torch.FloatTensor([exp['features'] for exp in batch])
                policies = torch.FloatTensor([exp['policy'] for exp in batch])
                values = torch.FloatTensor([exp['result'] for exp in batch])
                
                if self.config.device == "cuda":
                    features, policies, values = features.cuda(), policies.cuda(), values.cuda()
                
                # Forward pass
                pred_policies, pred_values = self.model(features)
                
                # Calculate losses
                policy_loss = nn.CrossEntropyLoss()(pred_policies, policies)
                value_loss = nn.MSELoss()(pred_values.squeeze(), values)
                loss = policy_loss + value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"Training loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_model(self):
        """Save trained model as .nnue format."""
        # Save PyTorch checkpoint
        checkpoint_path = f"{self.config.checkpoint_dir}/model_iter_{self.iteration}.pt"
        torch.save({
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss
        }, checkpoint_path)
        
        # Export to NNUE format for engine use
        export_nnue(self.model, self.config.model_output_path)
        self.logger.info(f"Model exported to {self.config.model_output_path}")
    
    def log_progress(self):
        """Log training progress."""
        self.logger.info(
            f"Iter {self.iteration:4d} | "
            f"Buffer: {len(self.exp_buffer):6d} | "
            f"Best Loss: {self.best_loss:.4f}"
        )

def main():
    """Main training entry point."""
    config = RLConfig()
    
    print("=== FluxFish RL Training (C++ Backend) ===")
    print(f"C++ Engine: {config.cpp_engine_path}")
    print(f"Output Model: {config.model_output_path}")
    print(f"Device: {config.device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Games/Iteration: {config.games_per_iteration}")
    print(f"Max Iterations: {config.max_iterations}")
    
    # Check C++ engine exists
    if not os.path.exists(config.cpp_engine_path):
        print(f"❌ C++ engine not found at {config.cpp_engine_path}")
        print("Please build the C++ engine first with 'make' in cpp/ directory")
        return
    
    # Check fluxfish.bin exists
    if not os.path.exists("fluxfish.bin"):
        print("❌ fluxfish.bin not found")
        print("Please ensure fluxfish.bin is in the current directory")
        return
    
    trainer = RLTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
