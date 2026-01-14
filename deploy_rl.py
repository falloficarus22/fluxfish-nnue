#!/usr/bin/env python3
"""
Deploy RL-trained model to UCI interface.
"""

import os
import shutil
import torch

def deploy_rl_model():
    """Deploy RL-trained model to UCI interface."""
    
    rl_model_path = "fluxfish_rl.nnue"
    target_model_path = "fluxfish.nnue"
    cpp_model_path = "fluxfish.bin"
    
    print("=== FluxFish RL Model Deployment ===")
    
    # Check if C++ binary exists (auto-exported during training)
    if os.path.exists(cpp_model_path):
        print(f"✅ C++ binary found: {cpp_model_path}")
        print("   This was automatically exported during RL training")
        
        # Copy to main model name for UCI
        shutil.copy2(cpp_model_path, "fluxfish_main.bin")
        print(f"✅ Copied to fluxfish_main.bin for UCI deployment")
        
        # Also copy PyTorch model for compatibility
        if os.path.exists(rl_model_path):
            shutil.copy2(rl_model_path, target_model_path)
            print(f"✅ Copied PyTorch model to {target_model_path}")
        
        print("\n🎯 RL model deployed successfully!")
        print("   Ready for UCI testing and Lichess deployment")
        print("\nTest with:")
        print("   python uci_cpp.py")
        return True
    
    # Fallback: Try to export from PyTorch model
    elif os.path.exists(rl_model_path):
        print(f"PyTorch model found: {rl_model_path}")
        print("Exporting to C++ binary format...")
        
        try:
            from export_model import export_nnue
            export_nnue(rl_model_path, cpp_model_path)
            print(f"✅ Exported to {cpp_model_path}")
            
            # Copy to main model name
            shutil.copy2(cpp_model_path, "fluxfish_main.bin")
            print(f"✅ Copied to fluxfish_main.bin")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
    else:
        print(f"❌ No RL model found: {rl_model_path}")
        print("Please run training first: python train_rl.py")
        return False

def main():
    deploy_rl_model()

if __name__ == "__main__":
    import time
    main()
