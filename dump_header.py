import struct
import numpy as np

def dump_nnue(filename):
    with open(filename, 'rb') as f:
        v = struct.unpack('<I', f.read(4))[0]
        h = struct.unpack('<I', f.read(4))[0]
        l = struct.unpack('<I', f.read(4))[0]
        d = f.read(l).decode('utf-8')
        print(f"Header Version: {v}")
        print(f"Header Hash: {h}")
        print(f"Description: {d}")
        
        # After description, there might be another hash for the FeatureTransformer
        ft_hash = struct.unpack('<I', f.read(4))[0]
        print(f"FT Hash: {ft_hash}")
        
        # Let's see if we can read the FT biases
        # If H=768, biases are 768 * 2 = 1536 bytes
        # If H=512, biases are 1024 bytes
        # Common H values: 256, 512, 768, 1024, 1536
        pos = f.tell()
        print(f"Current Position: {pos}")
        
        # Check next few bytes
        preview = f.read(64)
        print(f"Preview (hex): {preview.hex()}")

dump_nnue('nn-1c0000000000.nnue')
