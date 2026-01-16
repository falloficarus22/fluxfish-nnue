import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.engine
import numpy as np
import random
import os
import time
from multiprocessing import Pool, cpu_count
from nnue_model import FluxFishNNUE

# ============ CONFIGURATION ============
STOCKFISH_PATH = "/usr/games/stockfish"  # Ensure this path is correct!
BATCH_SIZE = 4096      # Efficient batch size
LEARNING_RATE = 0.001
NUM_POSITIONS = 2500000  # 2.5 Million positions for GM strength
EPOCHS = 35              # Train longer
NUM_WORKERS = max(1, cpu_count()) # Use all cores
SAVE_PATH = "fluxfish.nnue"
DATASET_FILE = "grandmaster_data.txt"

# Stockfish settings (Grandmaster Quality)
SF_DEPTH = 12       # Depth 12 is much better than 10
SF_TIME_LIMIT = 0.05 
# =======================================

class ChessDataset(Dataset):
    """Dataset of chess positions with Stockfish evaluations."""
    
    def __init__(self, positions_file: str = None):
        self.positions = []  # List of (fen, eval) tuples
        self.model = FluxFishNNUE()  # For feature extraction only
        
        if positions_file and os.path.exists(positions_file):
            self.load(positions_file)
    
    def add_position(self, fen: str, evaluation: float):
        """Add a position with its evaluation."""
        self.positions.append((fen, evaluation))
    
    def save(self, path: str):
        """Save dataset to file."""
        with open(path, 'w') as f:
            for fen, ev in self.positions:
                f.write(f"{fen}|{ev}\n")
        print(f"Saved {len(self.positions)} positions to {path}")
    
    def load(self, path: str):
        """Load dataset from file."""
        print(f"Loading dataset from {path}...")
        count = 0
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    self.positions.append((parts[0], float(parts[1])))
                    count += 1
        print(f"Loaded {count} positions.")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        fen, evaluation = self.positions[idx]
        board = chess.Board(fen)
        
        w_feat = self.model.get_features(board, chess.WHITE).float()
        b_feat = self.model.get_features(board, chess.BLACK).float()
        stm = torch.tensor([1.0 if board.turn == chess.WHITE else 0.0], dtype=torch.float32)
        label = torch.tensor([evaluation], dtype=torch.float32)
        
        return w_feat, b_feat, stm, label


def generate_random_position(depth: int = 10) -> chess.Board:
    """Generate a random legal position by playing random moves."""
    board = chess.Board()
    moves_played = random.randint(8, 60) # Deeper games for more realistic positions
    
    for _ in range(moves_played):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    
    return board


def generate_position_batch(args) -> list:
    """Generate a batch of positions with evaluations (for multiprocessing)."""
    batch_size, worker_id = args
    results = []
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        
        for i in range(batch_size):
            board = generate_random_position(random.randint(10, 80))
            
            if board.is_game_over():
                continue
            
            # Get Stockfish evaluation
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
                score = info["score"].relative
                
                if score.is_mate():
                    ev = 1.0 if score.mate() > 0 else -1.0
                else:
                    cp = max(-2000, min(2000, score.score()))
                    ev = np.tanh(cp / 400.0) # Squeeze to [-1, 1]
                
                results.append((board.fen(), ev))
                
            except:
                continue
        
        engine.quit()
        
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
    
    return results


def generate_dataset_parallel(target_positions: int, output_file: str):
    """Generate training data using multiple processes."""
    
    # Check if we have existing data to resume from
    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_count = sum(1 for _ in f)
        print(f"Found existing dataset with {existing_count} positions. Resuming...")
    
    remaining = target_positions - existing_count
    if remaining <= 0:
        print("Dataset already complete!")
        return ChessDataset(output_file)

    print(f"Generating {remaining} new positions using {NUM_WORKERS} workers...")
    
    # Generate in chunks to save progress safely
    CHUNK_SIZE = 5000 * NUM_WORKERS
    
    dataset = ChessDataset(output_file) if existing_count > 0 else ChessDataset()
    
    while len(dataset) < target_positions:
        start_time = time.time()
        
        current_chunk = min(CHUNK_SIZE, target_positions - len(dataset))
        positions_per_worker = current_chunk // NUM_WORKERS
        work_items = [(positions_per_worker, i) for i in range(NUM_WORKERS)]
        
        with Pool(NUM_WORKERS) as pool:
            results = pool.map(generate_position_batch, work_items)
            
        new_positions = 0
        with open(output_file, 'a') as f:
            for batch in results:
                for fen, ev in batch:
                    dataset.add_position(fen, ev)
                    f.write(f"{fen}|{ev}\n")
                    new_positions += 1
        
        elapsed = time.time() - start_time
        rate = new_positions / elapsed if elapsed > 0 else 0
        print(f"Progress: {len(dataset)}/{target_positions} (+{new_positions} @ {rate:.1f} pos/s)")
        
    return dataset


def train_model(dataset: ChessDataset, epochs: int = EPOCHS):
    """Train the NNUE model on the dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = FluxFishNNUE().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  
        pin_memory=True
    )
    
    print(f"\nTraining on {len(dataset)} positions for {epochs} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, (w_feat, b_feat, stm, labels) in enumerate(dataloader):
            w_feat = w_feat.to(device)
            b_feat = b_feat.to(device)
            stm = stm.to(device).squeeze(-1).bool()
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = list()
            # Simple batched forward (naive loop if model doesn't support batching yet)
            # Assuming nnue_model is updated or we use loop:
            # Let's use the loop in this script to be safe unless we are sure about model support
            # Actually, standard PyTorch model should handle batches if written correctly.
            # But earlier we saw `nnue_model.py` had scalar assumptions or loops.
            # Let's assume user updated it or use loop for safety if not.
            # The previous turn updated nnue_model.py to support batches, so we can try:
            try:
                # Optimized batch call
                outputs = model(w_feat, b_feat, stm)
            except:
                # Fallback loop
                out_list = []
                for i in range(len(w_feat)):
                   out_list.append(model(w_feat[i], b_feat[i], stm[i].item())) 
                outputs = torch.stack(out_list)

            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.5f}, Time: {elapsed:.1f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save strictly state_dict for export compatibility
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  Saved best model (loss: {avg_loss:.5f})")
    
    print(f"\nTraining complete! Best loss: {best_loss:.5f}")
    print(f"Model saved to: {SAVE_PATH}")
    return model


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("FluxFish Grandmaster Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load existing training data only
    if not os.path.exists(DATASET_FILE):
        print(f"ERROR: {DATASET_FILE} not found!")
        print("Please ensure grandmaster_data.txt exists before training.")
        return
    
    dataset = ChessDataset(DATASET_FILE)
    print(f"Loaded {len(dataset)} positions from {DATASET_FILE}")
    
    # Step 2: Train the model
    if len(dataset) < 1000:
        print("ERROR: Not enough training data!")
        return
    
    train_model(dataset, epochs=EPOCHS)
    
    print("\n✓ Training complete! Run export_model.py next.")

if __name__ == "__main__":
    main()
