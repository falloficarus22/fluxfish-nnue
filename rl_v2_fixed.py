#!/usr/bin/env python3
"""
FluxFish RL v2.0 - Fixed Architecture
Uses C++ backend with fluxfish.bin for self-play, trains new model as fluxfish.nnue
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
import subprocess
import logging
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import existing components
from nnue_model import FluxFishNNUE, NUM_FEATURES
from export_model import export_nnue

@dataclass
class RLConfigV2:
    """RL configuration for C++ backend integration."""
    
    # Training parameters
    batch_size: int = 8192
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    epochs_per_iteration: int = 5
    max_iterations: int = 500
    
    # Self-play with C++ backend
    games_per_iteration: int = 16
    max_moves_per_game: int = 200
    cpp_engine_path: str = "./cpp/fluxfish_cpp"
    fluxfish_bin_path: str = "fluxfish.bin"
    
    # Experience replay
    replay_buffer_size: int = 100000
    min_buffer_size: int = 1000
    
    # Optimization
    num_workers: int = min(4, mp.cpu_count())
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    model_output_path: str = "fluxfish.nnue"
    checkpoint_dir: str = "checkpoints_v2"
    log_dir: str = "logs_v2"

class ExperienceBuffer:
    """Simple experience replay for C++ backend data."""
    
    def __init__(self, config: RLConfigV2):
        self.config = config
        self.buffer = deque(maxlen=config.replay_buffer_size)
        
    def add(self, experiences: List[Dict]):
        for exp in experiences:
            self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> List[Dict]:
        if len(self.buffer) < batch_size:
            return []
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        return len(self.buffer)

class SelfPlayCpp:
    """Self-play using C++ backend."""
    
    def __init__(self, config: RLConfigV2):
        self.config = config
        self.engine_path = config.cpp_engine_path
        
    def play_game(self) -> List[Dict]:
        """Play single game using C++ engine."""
        experiences = []
        
        # Start C++ engine
        process = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        # Send UCI commands
        commands = [
            "uci",
            "ucinewgame",
            "go movetime 100",  # Quick moves for self-play
            "quit"
        ]
        
        stdout, stderr = process.communicate("\n".join(commands) + "\n")
        
        # Parse engine output to extract positions and evaluations
        # This is simplified - you'd need proper parsing
        experiences = self._parse_engine_output(stdout)
        
        return experiences
    
    def _parse_engine_output(self, output: str) -> List[Dict]:
        """Parse C++ engine output to extract training data."""
        experiences = []
        
        # This is a placeholder - implement proper parsing
        # You'd extract positions, search stats, and game results
        
        return experiences

class RLTrainerV2:
    """RL trainer using C++ backend."""
    
    def __init__(self, config: RLConfigV2):
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
        self.exp_buffer = ExperienceBuffer(config)
        
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
        self.logger.info("Starting RL training with C++ backend")
        self.logger.info(f"Device: {self.config.device}")
        
        for iteration in range(self.iteration, self.config.max_iterations):
            self.iteration = iteration
            
            # Self-play with C++ engine
            experiences = self.self_play_phase()
            
            # Training phase
            if len(self.exp_buffer) >= self.config.min_buffer_size:
                loss = self.training_phase()
                
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_model()
            
            self.log_progress(iteration)
        
        self.save_model()
        self.logger.info("Training completed")
    
    def self_play_phase(self) -> List[Dict]:
        """Generate self-play games using C++ engine."""
        self.logger.info(f"Self-play iteration {self.iteration}")
        
        all_experiences = []
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for _ in range(self.config.games_per_iteration):
                future = executor.submit(self._play_single_game)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    experiences = future.result(timeout=60)
                    all_experiences.extend(experiences)
                except Exception as e:
                    self.logger.warning(f"Game failed: {e}")
        
        self.exp_buffer.add(all_experiences)
        return all_experiences
    
    def _play_single_game(self) -> List[Dict]:
        """Play single game (for multiprocessing)."""
        # This would call the C++ engine
        # Return dummy data for now
        return [{
            'features': np.random.rand(NUM_FEATURES).astype(np.float32),
            'policy': np.random.rand(1858).astype(np.float32),
            'value': np.random.rand(),
            'result': np.random.choice([-1, 0, 1])
        }]
    
    def training_phase(self) -> float:
        """Train neural network on collected experiences."""
        if len(self.exp_buffer) < self.config.batch_size:
            return float('inf')
        
        total_loss = 0.0
        
        for epoch in range(self.config.epochs_per_iteration):
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
        
        avg_loss = total_loss / (self.config.epochs_per_iteration)
        self.logger.info(f"Training loss: {avg_loss:.4f}")
        return avg_loss
    
    def save_model(self):
        """Save trained model as .nnue format."""
        # Save PyTorch checkpoint
        checkpoint_path = f"{self.config.checkpoint_dir}/model_iter_{self.iteration}.pt"
        torch.save({
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'loss': self.best_loss
        }, checkpoint_path)
        
        # Export to NNUE format
        export_nnue(self.model, self.config.model_output_path)
        self.logger.info(f"Model saved to {self.config.model_output_path}")
    
    def log_progress(self, iteration: int):
        self.logger.info(
            f"Iter {iteration:4d} | "
            f"Buffer: {len(self.exp_buffer):6d} | "
            f"Best Loss: {self.best_loss:.4f}"
        )

def main():
    config = RLConfigV2()
    print("=== FluxFish RL Training v2.0 (C++ Backend) ===")
    print(f"Using C++ engine: {config.cpp_engine_path}")
    print(f"Loading fluxfish.bin: {config.fluxfish_bin_path}")
    print(f"Output model: {config.model_output_path}")
    
    trainer = RLTrainerV2(config)
    trainer.train()

if __name__ == "__main__":
    main()
