#!/usr/bin/env python3
"""
Test script for FluxFish RL v2.0
Verifies the new RL system components work correctly.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from rl_v2 import RLConfigV2, RLTrainerV2, ExperienceBuffer, SelfPlayGame
        from nnue_model import FluxFishNNUE, NUM_FEATURES
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration system."""
    print("\nTesting configuration...")
    try:
        config = RLConfigV2()
        print(f"✅ Default config created")
        print(f"  - Device: {config.device}")
        print(f"  - Batch size: {config.batch_size}")
        print(f"  - Buffer size: {config.replay_buffer_size:,}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_experience_buffer():
    """Test experience replay buffer."""
    print("\nTesting experience buffer...")
    try:
        config = RLConfigV2()
        config.replay_buffer_size = 1000  # Smaller for testing
        buffer = ExperienceBuffer(config)
        
        # Add some dummy experiences
        dummy_experiences = [
            {
                'features': np.random.rand(NUM_FEATURES).astype(np.float32),
                'policy': np.random.rand(1858).astype(np.float32),
                'value': np.random.rand(),
                'result': np.random.choice([-1, 0, 1])
            }
            for _ in range(10)
        ]
        
        buffer.add(dummy_experiences)
        print(f"✅ Added {len(dummy_experiences)} experiences")
        print(f"  - Buffer size: {len(buffer)}")
        
        # Test sampling
        if len(buffer) >= 5:
            experiences, indices, weights = buffer.sample(5)
            print(f"✅ Sampled {len(experiences)} experiences")
            print(f"  - Sample weights shape: {weights.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Experience buffer error: {e}")
        return False

def test_model():
    """Test NNUE model."""
    print("\nTesting NNUE model...")
    try:
        from nnue_model import FluxFishNNUE
        
        model = FluxFishNNUE()
        print(f"✅ Model created")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 4
        dummy_features = torch.randn(batch_size, NUM_FEATURES)
        
        with torch.no_grad():
            policy, value = model(dummy_features)
        
        print(f"✅ Forward pass successful")
        print(f"  - Policy shape: {policy.shape}")
        print(f"  - Value shape: {value.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

def test_trainer_initialization():
    """Test trainer initialization."""
    print("\nTesting trainer initialization...")
    try:
        config = RLConfigV2()
        config.max_iterations = 5  # Small for testing
        config.games_per_iteration = 2
        config.batch_size = 8
        
        trainer = RLTrainerV2(config)
        print(f"✅ Trainer initialized")
        print(f"  - Device: {config.device}")
        print(f"  - Model on device: {next(trainer.model.parameters()).device}")
        
        return True
    except Exception as e:
        print(f"❌ Trainer initialization error: {e}")
        return False

def test_directories():
    """Test directory creation."""
    print("\nTesting directory creation...")
    try:
        config = RLConfigV2()
        
        # Test directory creation
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        print(f"✅ Directories created")
        print(f"  - Checkpoint dir: {config.checkpoint_dir}")
        print(f"  - Log dir: {config.log_dir}")
        
        # Check they exist
        assert Path(config.checkpoint_dir).exists()
        assert Path(config.log_dir).exists()
        
        return True
    except Exception as e:
        print(f"❌ Directory error: {e}")
        return False

def test_device_setup():
    """Test device setup and CUDA availability."""
    print("\nTesting device setup...")
    try:
        config = RLConfigV2()
        
        print(f"✅ Device configuration")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        print(f"  - Selected device: {config.device}")
        print(f"  - Mixed precision: {config.mixed_precision}")
        
        if torch.cuda.is_available():
            print(f"  - CUDA device: {torch.cuda.get_device_name()}")
            print(f"  - CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return True
    except Exception as e:
        print(f"❌ Device setup error: {e}")
        return False

def main():
    """Run all tests."""
    print("=== FluxFish RL v2.0 Test Suite ===\n")
    
    tests = [
        test_imports,
        test_config,
        test_device_setup,
        test_model,
        test_experience_buffer,
        test_directories,
        test_trainer_initialization,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing
    
    print("=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! RL v2.0 is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python train_rl_v2.py' to start training")
        print("2. Choose 'Quick Test' to verify everything works")
        print("3. Progress to longer training sessions")
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
