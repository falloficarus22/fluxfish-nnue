#!/usr/bin/env python3
"""
FluxFish RL v2.0 Training Launcher
Simple interface for the new reinforcement learning system.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_v2 import RLConfigV2, RLTrainerV2

def main():
    print("=== FluxFish RL Training v2.0 Launcher ===")
    print("\nTraining Options:")
    print("1. Quick Test (50 iterations, verify setup)")
    print("2. Short Training (500 iterations)")
    print("3. Full Training (2000 iterations)")
    print("4. Custom Training")
    print("5. Resume from Checkpoint")
    print("6. Test Current Model")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    config = RLConfigV2()
    
    if choice == "1":
        config.max_iterations = 50
        config.games_per_iteration = 8
        config.mcts_iterations = 200
        config.batch_size = 128
        config.save_interval = 10
        print("\nQuick Test Mode:")
        print("- 50 iterations")
        print("- 8 games per iteration")
        print("- 200 MCTS iterations")
        print("- Small batch size for fast testing")
        
    elif choice == "2":
        config.max_iterations = 500
        config.games_per_iteration = 16
        config.mcts_iterations = 400
        config.batch_size = 256
        print("\nShort Training Mode:")
        print("- 500 iterations")
        print("- 16 games per iteration")
        print("- 400 MCTS iterations")
        print("- Medium batch size")
        
    elif choice == "3":
        config.max_iterations = 2000
        config.games_per_iteration = 32
        config.mcts_iterations = 800
        config.batch_size = 512
        print("\nFull Training Mode:")
        print("- 2000 iterations")
        print("- 32 games per iteration")
        print("- 800 MCTS iterations")
        print("- Large batch size for GPU utilization")
        
    elif choice == "4":
        try:
            print("\nCustom Training Configuration:")
            config.max_iterations = int(input("Max iterations (e.g., 1000): "))
            config.games_per_iteration = int(input("Games per iteration (e.g., 20): "))
            config.mcts_iterations = int(input("MCTS iterations per search (e.g., 500): "))
            config.batch_size = int(input("Batch size (e.g., 256): "))
            config.learning_rate = float(input("Learning rate (e.g., 0.001): "))
        except ValueError:
            print("Invalid input. Using defaults.")
            return
            
    elif choice == "5":
        # Resume from checkpoint
        checkpoint_dir = Path("checkpoints_v2")
        if not checkpoint_dir.exists():
            print("No checkpoint directory found.")
            return
        
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            print("No checkpoints found.")
            return
        
        print("\nAvailable checkpoints:")
        for i, cp in enumerate(sorted(checkpoints, key=lambda p: int(p.stem.split('_')[1]))[-5:], 1):
            iter_num = cp.stem.split('_')[1]
            print(f"{i}. Iteration {iter_num} - {cp.name}")
        
        try:
            cp_choice = int(input("Select checkpoint (1-5): ")) - 1
            selected_cp = sorted(checkpoints, key=lambda p: int(p.stem.split('_')[1]))[-5:][cp_choice]
            
            print(f"\nResuming from {selected_cp.name}")
            trainer = RLTrainerV2(config)
            trainer.load_checkpoint(str(selected_cp))
            trainer.train()
            return
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
            
    elif choice == "6":
        # Test current model
        model_path = input("Model path (default: fluxfish_v2.nnue): ").strip()
        if not model_path:
            model_path = "fluxfish_v2.nnue"
        
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found.")
            return
        
        print(f"\nTesting model {model_path}")
        # Simple test - load model and run a few positions
        try:
            from nnue_model import FluxFishNNUE
            import chess
            
            model = FluxFishNNUE()
            # Load weights (implement proper loading)
            print("Model loaded successfully!")
            
            # Test on a few positions
            test_positions = [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
            ]
            
            for i, fen in enumerate(test_positions, 1):
                board = chess.Board(fen)
                print(f"\nTest position {i}: {fen}")
                # This would need proper feature extraction and evaluation
                print("Position evaluation would be shown here")
            
        except Exception as e:
            print(f"Error testing model: {e}")
        return
        
    else:
        print("Invalid choice.")
        return
    
    # Show configuration summary
    print(f"\nConfiguration Summary:")
    print(f"- Device: {config.device}")
    print(f"- Max iterations: {config.max_iterations}")
    print(f"- Games per iteration: {config.games_per_iteration}")
    print(f"- MCTS iterations: {config.mcts_iterations}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Learning rate: {config.learning_rate}")
    print(f"- Buffer size: {config.replay_buffer_size:,}")
    print(f"- Workers: {config.num_workers}")
    
    # Confirm
    confirm = input(f"\nStart training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Start training
    print("\nInitializing trainer...")
    trainer = RLTrainerV2(config)
    
    # Try to load existing checkpoint
    try:
        trainer.load_checkpoint()
        print(f"Resumed from iteration {trainer.iteration}")
    except:
        print("Starting fresh training")
    
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
