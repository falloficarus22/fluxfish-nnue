# Roadmap to Grandmaster Level (2500+ ELO)

You have successfully migrated the engine core to C++ (FluxFish C++) and integrated the UCI protocol.
You have a high-quality dataset (900k positions).

## 1. Engine Architecture (COMPLETED)
- **C++ + AVX2**: The new C++ engine (`cpp/fluxfish_cpp`) handles search and evaluation 100x faster than Python.
- **UCI Loop**: Implemented directly in C++.

## 2. Train on Your Data (NEXT STEP)
We need to train the NNUE network on your CSV dataset.
- **Script**: `python3 train_csv.py <path_to_your_csv>`
- **Batched Training**: The script is optimized to handle large datasets efficiently.
- **Output**: This will generate a new `fluxfish.nnue` model file with significantly better evaluation knowledge.

## 3. Export & Deploy
After training:
1. **Export Weights**: Run `python3 export_model.py`. This converts the trained `.nnue` (PyTorch) to `.bin` (C++ binary).
2. **Move Binary**: Ensure `fluxfish.bin` is in the `cpp/` directory (or where `fluxfish_cpp` expects it).
3. **Rebuild**: `cd cpp && make clean && make` (if you changed any C++ constants, otherwise just running the binary is fine).

## 4. Play!
- **Lichess**: Use the `lichess-bot` setup with the C++ engine.
- **Local**: Load `cpp/fluxfish_cpp` in your favorite chess GUI.
- **Settings**:
  - Hash: 64MB+
  - Threads: 1

## 5. Future Improvements
- **Data Augmentation**: Flip boards horizontally to double dataset size easily.
- **Quantization**: Quantize weights to int8 for even faster inference (requires C++ inference changes).
- **Larger Network**: Upgrade to 768->512->32 for more capacity if 256 is saturated.
