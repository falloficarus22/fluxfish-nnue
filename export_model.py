import torch
import numpy as np
import struct
from nnue_model import FluxFishNNUE

def export_nnue(model_path, export_path):
    # Load the model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = FluxFishNNUE()
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    with open(export_path, 'wb') as f:
        # 1. Feature Transformer (self.ft)
        # Weights: (256, 768), Bias: (256,)
        ft_weight = model.ft.weight.data.numpy()  # [Hidden, Input]
        ft_bias = model.ft.bias.data.numpy()      # [Hidden]
        f.write(ft_weight.astype(np.float32).tobytes())
        f.write(ft_bias.astype(np.float32).tobytes())

        # 2. Layer 1 (self.l1)
        # Weights: (32, 512), Bias: (32,)
        l1_weight = model.l1.weight.data.numpy()
        l1_bias = model.l1.bias.data.numpy()
        f.write(l1_weight.astype(np.float32).tobytes())
        f.write(l1_bias.astype(np.float32).tobytes())

        # 3. Layer 2 (self.l2)
        # Weights: (32, 32), Bias: (32,)
        l2_weight = model.l1.weight.data.numpy() # Note: Your model had l2, let me double check
        # Checking your nnue_model.py... ah, l2 is (32, 32). 
        # Wait, in your nnue_model.py, L1 is (32, 512), L2 is (32, 32), L3 is (1, 32)
        l2_weight = model.l2.weight.data.numpy()
        l2_bias = model.l2.bias.data.numpy()
        f.write(l2_weight.astype(np.float32).tobytes())
        f.write(l2_bias.astype(np.float32).tobytes())

        # 4. Layer 3 (self.l3)
        # Weights: (1, 32), Bias: (1,)
        l3_weight = model.l3.weight.data.numpy()
        l3_bias = model.l3.bias.data.numpy()
        f.write(l3_weight.astype(np.float32).tobytes())
        f.write(l3_bias.astype(np.float32).tobytes())

    print(f"Successfully exported NNUE weights to {export_path}")

if __name__ == "__main__":
    export_nnue("fluxfish.nnue", "fluxfish.bin")
