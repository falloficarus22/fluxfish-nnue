import torch
import numpy as np
import os
from nnue_model import FluxFishNNUE

def import_nnue(bin_path, nnue_path):
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found.")
        return

    model = FluxFishNNUE()
    
    with open(bin_path, 'rb') as f:
        # Load weights and biases in the same order as export_model.py
        
        # 1. Feature Transformer (self.ft)
        # Weights: (256, 768), Bias: (256,)
        ft_weight = np.frombuffer(f.read(256 * 768 * 4), dtype=np.float32).reshape(256, 768)
        ft_bias = np.frombuffer(f.read(256 * 4), dtype=np.float32)
        model.ft.weight.data = torch.from_numpy(ft_weight)
        model.ft.bias.data = torch.from_numpy(ft_bias)

        # 2. Layer 1 (self.l1)
        # Weights: (32, 512), Bias: (32,)
        l1_weight = np.frombuffer(f.read(32 * 512 * 4), dtype=np.float32).reshape(32, 512)
        l1_bias = np.frombuffer(f.read(32 * 4), dtype=np.float32)
        model.l1.weight.data = torch.from_numpy(l1_weight)
        model.l1.bias.data = torch.from_numpy(l1_bias)

        # 3. Layer 2 (self.l2)
        # Weights: (32, 32), Bias: (32,)
        l2_weight = np.frombuffer(f.read(32 * 32 * 4), dtype=np.float32).reshape(32, 32)
        l2_bias = np.frombuffer(f.read(32 * 4), dtype=np.float32)
        model.l2.weight.data = torch.from_numpy(l2_weight)
        model.l2.bias.data = torch.from_numpy(l2_bias)

        # 4. Layer 3 (self.l3)
        # Weights: (1, 32), Bias: (1,)
        l3_weight = np.frombuffer(f.read(1 * 32 * 4), dtype=np.float32).reshape(1, 32)
        l3_bias = np.frombuffer(f.read(1 * 4), dtype=np.float32)
        model.l3.weight.data = torch.from_numpy(l3_weight)
        model.l3.bias.data = torch.from_numpy(l3_bias)

    # Save as PyTorch state dict
    torch.save(model.state_dict(), nnue_path)
    print(f"Successfully converted {bin_path} to {nnue_path}")

if __name__ == "__main__":
    import_nnue("fluxfish.bin", "fluxfish.nnue")
