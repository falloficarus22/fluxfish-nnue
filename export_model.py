import torch
import numpy as np
import struct
from nnue_model import FluxFishNNUE

def export_nnue(model, export_path):
    """
    Export NNUE model to binary format.
    Args:
        model: Either a model object or path to model file
        export_path: Path to save the exported model
    """
    # Handle both model object and file path
    if isinstance(model, str):
        # Load the model from file
        checkpoint = torch.load(model, map_location='cpu')
        model_obj = FluxFishNNUE()
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_obj.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_obj.load_state_dict(checkpoint)
        model_obj.eval()
    else:
        # Use the model object directly
        model_obj = model
        model_obj.eval()

    with open(export_path, 'wb') as f:
        # 1. Feature Transformer (self.ft)
        # Weights: (256, 768), Bias: (256,)
        ft_weight = model_obj.ft.weight.data.cpu().numpy()  # Move to CPU before numpy
        ft_bias = model_obj.ft.bias.data.cpu().numpy()      # Move to CPU before numpy
        f.write(ft_weight.astype(np.float32).tobytes())
        f.write(ft_bias.astype(np.float32).tobytes())

        # 2. Layer 1 (self.l1)
        # Weights: (32, 512), Bias: (32,)
        l1_weight = model_obj.l1.weight.data.cpu().numpy()
        l1_bias = model_obj.l1.bias.data.cpu().numpy()
        f.write(l1_weight.astype(np.float32).tobytes())
        f.write(l1_bias.astype(np.float32).tobytes())

        # 3. Layer 2 (self.l2)
        # Weights: (32, 32), Bias: (32,)
        l2_weight = model_obj.l2.weight.data.cpu().numpy()
        l2_bias = model_obj.l2.bias.data.cpu().numpy()
        f.write(l2_weight.astype(np.float32).tobytes())
        f.write(l2_bias.astype(np.float32).tobytes())

        # 4. Layer 3 (self.l3)
        # Weights: (1, 32), Bias: (1,)
        l3_weight = model_obj.l3.weight.data.cpu().numpy()
        l3_bias = model_obj.l3.bias.data.cpu().numpy()
        f.write(l3_weight.astype(np.float32).tobytes())
        f.write(l3_bias.astype(np.float32).tobytes())

    print(f"Successfully exported NNUE weights to {export_path}")

if __name__ == "__main__":
    export_nnue("fluxfish.nnue", "fluxfish.bin")
