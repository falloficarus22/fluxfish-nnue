#!/usr/bin/env python3
"""
Test C++ backend integration for RL training
"""

import subprocess
import os

def test_cpp_engine():
    """Test that C++ engine is accessible."""
    print("Testing C++ engine...")
    
    engine_path = "./cpp/fluxfish_cpp"
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found at {engine_path}")
        return False
    
    try:
        # Test engine with simple UCI commands
        process = subprocess.Popen(
            ["wsl", "-d", "Ubuntu", engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        # Send UCI commands
        stdout, stderr = process.communicate("uci\nquit\n", timeout=10)
        
        if "id name FluxFish C++" in stdout:
            print("✅ C++ engine responding correctly")
            print(f"Engine output: {stdout[:200]}...")
            return True
        else:
            print(f"❌ Unexpected engine output: {stdout}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing engine: {e}")
        return False

def test_fluxfish_bin():
    """Test fluxfish.bin exists."""
    print("\nTesting fluxfish.bin...")
    
    if os.path.exists("fluxfish.bin"):
        print("✅ fluxfish.bin found")
        return True
    else:
        print("❌ fluxfish.bin not found")
        return False

def test_directory_structure():
    """Test required directories."""
    print("\nTesting directory structure...")
    
    dirs_to_check = ["cpp", "checkpoints_rl", "logs_rl"]
    all_exist = True
    
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing")
            all_exist = False
    
    return all_exist

def main():
    print("=== C++ Backend Integration Test ===\n")
    
    tests = [
        test_directory_structure,
        test_fluxfish_bin,
        test_cpp_engine
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=== Test Results ===")
    print(f"Passed: {passed}/3")
    
    if passed == 3:
        print("🎉 All tests passed! Ready for RL training.")
        print("\nTo start training:")
        print("1. Ensure dependencies are installed in WSL")
        print("2. Run: python rl_train_final.py")
    else:
        print("⚠️  Some tests failed. Fix issues before training.")

if __name__ == "__main__":
    main()
