#!/usr/bin/env python3
"""
Debug RL training pipeline to identify issues.
"""

import torch
import chess
import numpy as np
from nnue_model import FluxFishNNUE
from fast_selfplay import FastSelfPlay

def test_model_forward():
    """Test model forward pass."""
    print("=== Testing Model Forward Pass ===")
    
    model = FluxFishNNUE()
    model.eval()
    
    # Test with start position
    board = chess.Board()
    
    try:
        # Get features
        white_features = model.get_features(board, chess.WHITE)
        black_features = model.get_features(board, chess.BLACK)
        stm = board.turn == chess.WHITE
        
        print(f"White features shape: {white_features.shape}")
        print(f"Black features shape: {black_features.shape}")
        print(f"Side to move: {stm}")
        
        # Forward pass
        with torch.no_grad():
            value = model(white_features.unsqueeze(0), black_features.unsqueeze(0), stm)
            print(f"Model output: {value.item():.6f}")
            print("✅ Model forward pass works")
            
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False
    
    return True

def test_fast_selfplay():
    """Test fast self-play."""
    print("\n=== Testing Fast Self-Play ===")
    
    try:
        fast_play = FastSelfPlay()
        
        # Test single move
        board = chess.Board()
        move = fast_play.get_cpp_move(board.fen(), 100)
        print(f"C++ engine move: {move}")
        
        # Test game generation
        game_data = fast_play.play_fast_game(max_moves=10, mcts_iterations=100)
        print(f"Generated {len(game_data)} positions")
        
        if game_data:
            fen, value, policy = game_data[0]
            print(f"Sample FEN: {fen[:50]}...")
            print(f"Sample value: {value:.6f}")
            print(f"Policy length: {len(policy)}")
            print("✅ Fast self-play works")
            
    except Exception as e:
        print(f"❌ Fast self-play failed: {e}")
        return False
    
    return True

def test_training_step():
    """Test a single training step."""
    print("\n=== Testing Training Step ===")
    
    try:
        model = FluxFishNNUE()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy data
        board = chess.Board()
        white_features = model.get_features(board, chess.WHITE)
        black_features = model.get_features(board, chess.BLACK)
        stm = board.turn == chess.WHITE
        
        # Forward pass
        predicted = model(white_features.unsqueeze(0), black_features.unsqueeze(0), stm)
        target = torch.tensor([[0.1]])  # Dummy target
        
        # Compute loss
        loss = torch.nn.MSELoss()(predicted, target)
        print(f"Training loss: {loss.item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("✅ Training step works")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False
    
    return True

def main():
    """Run all debug tests."""
    print("🔍 FluxFish RL Training Debug")
    print("=" * 50)
    
    tests = [
        ("Model Forward Pass", test_model_forward),
        ("Fast Self-Play", test_fast_selfplay),
        ("Training Step", test_training_step),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! Training should work.")
    else:
        print("\n⚠️  Some tests failed. Fix issues before training.")

if __name__ == "__main__":
    main()
