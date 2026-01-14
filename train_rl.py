#!/usr/bin/env python3
"""
FluxFish RL Training Launcher
Simple interface to start and monitor training.
"""

import sys
import os
from rl_train import RLTrainer, RLConfig

def main():
    print("=== FluxFish RL Training Launcher ===")
    print("\nTraining Options:")
    print("1. Quick Training (10 iterations, fast)")
    print("2. Standard Training (100 iterations)")
    print("3. Full Training (1000 iterations)")
    print("4. Custom Training")
    print("5. Evaluate Current Model")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    config = RLConfig()
    
    if choice == "1":
        config.max_iterations = 10
        config.games_per_iteration = 5
        config.mcts_iterations = 200
        print("\nQuick Training Mode:")
        print("- 10 iterations")
        print("- 5 games per iteration")
        print("- 200 MCTS iterations")
        
    elif choice == "2":
        config.max_iterations = 100
        config.games_per_iteration = 10
        config.mcts_iterations = 300
        print("\nStandard Training Mode:")
        print("- 100 iterations")
        print("- 10 games per iteration")
        print("- 300 MCTS iterations")
        
    elif choice == "3":
        config.max_iterations = 1000
        config.games_per_iteration = 20
        config.mcts_iterations = 500
        print("\nFull Training Mode:")
        print("- 1000 iterations")
        print("- 20 games per iteration")
        print("- 500 MCTS iterations")
        
    elif choice == "4":
        try:
            config.max_iterations = int(input("Max iterations: "))
            config.games_per_iteration = int(input("Games per iteration: "))
            config.mcts_iterations = int(input("MCTS iterations per search: "))
            config.epochs_per_iteration = int(input("Epochs per iteration: "))
        except ValueError:
            print("Invalid input. Using defaults.")
            return
            
    elif choice == "5":
        from evaluate_rl import evaluate_model
        model_path = input("Model path (default: fluxfish_rl.nnue): ").strip()
        if not model_path:
            model_path = "fluxfish_rl.nnue"
        evaluate_model(model_path)
        return
        
    else:
        print("Invalid choice.")
        return
    
    # Confirm
    confirm = input(f"\nStart training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Start training
    print("\nStarting training...")
    trainer = RLTrainer(config)
    
    # Load checkpoint first to get current state
    trainer.load_checkpoint()
    current_iter = trainer.iteration
    
    # Set target based on training mode
    if choice == "1":  # Quick Training
        target_iterations = current_iter + 10
    elif choice == "2":  # Standard Training  
        target_iterations = current_iter + 100
    elif choice == "3":  # Full Training
        target_iterations = current_iter + 1000
    elif choice == "4":  # Custom
        target_iterations = config.max_iterations
    else:
        target_iterations = config.max_iterations
    
    trainer.config.max_iterations = target_iterations
    
    print(f"Current iteration: {current_iter}")
    print(f"Training until iteration: {target_iterations}")
    print(f"Will run {target_iterations - current_iter} more iterations")
    
    trainer.train()

if __name__ == "__main__":
    main()
