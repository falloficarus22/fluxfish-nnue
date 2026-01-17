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
import argparse
from multiprocessing import Pool, cpu_count
from nnue_model import FluxFishNNUE

# ============ CONFIGURATION ============
STOCKFISH_PATH = "/usr/games/stockfish"  # Ensure this path is correct!
BATCH_SIZE = 8192      # Larger batch size for GPU
LEARNING_RATE = 0.001
NUM_POSITIONS = 2500000  # 2.5 Million positions for GM strength
EPOCHS = 10              # Train longer
NUM_WORKERS = min(4, cpu_count()) # Reduced workers for GPU efficiency
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


def train_model(dataset: ChessDataset, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, 
                learning_rate: float = LEARNING_RATE, num_workers: int = NUM_WORKERS,
                device=None, save_path: str = SAVE_PATH, verbose: bool = False):
    """Train the NNUE model on the dataset."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    model = FluxFishNNUE().to(device)
    
    # Use mixed precision training for GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Create dataloader first for scheduler
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,  
        shuffle=True,
        num_workers=num_workers,  # Enable multiprocessing for data loading
        pin_memory=True,          # Faster GPU transfers
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2         # Prefetch batches
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.1  # Warmup for 10% of training
    )
    criterion = nn.MSELoss(reduction = 'mean'))
    
    print(f"\nTraining on {len(dataset)} positions for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, (w_feat, b_feat, stm, labels) in enumerate(dataloader):
            w_feat = w_feat.to(device, non_blocking=True)
            b_feat = b_feat.to(device, non_blocking=True)
            stm = stm.to(device).squeeze(-1).bool()
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Use mixed precision if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    try:
                        outputs = model(w_feat, b_feat, stm)
                    except:
                        # Fallback loop for batch compatibility
                        out_list = []
                        for i in range(len(w_feat)):
                           out_list.append(model(w_feat[i], b_feat[i], stm[i].item())) 
                        outputs = torch.stack(out_list)
                    
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU fallback
                try:
                    outputs = model(w_feat, b_feat, stm)
                except:
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

        # OneCycleLR handles stepping automatically per batch
        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time
        
        # Clear GPU cache periodically
        if device.type == 'cuda' and (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.5f}, LR: {scheduler.get_last_lr()[0]:.6f}, Time: {elapsed:.1f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save strictly state_dict for export compatibility
            torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (loss: {avg_loss:.5f})")
    
    print(f"\nTraining complete! Best loss: {best_loss:.5f}")
    print(f"Model saved to: {save_path}")
    print(f"Model saved to: {save_path}")
    return model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train FluxFish NNUE model on chess positions")
    
    # Data arguments
    parser.add_argument("--data", type=str, default=DATASET_FILE,
                       help=f"Path to training data file (default: {DATASET_FILE})")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                       help=f"Path to save trained model (default: {SAVE_PATH})")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help=f"Batch size for training (default: {BATCH_SIZE})")
    parser.add_argument("--lr", "--learning-rate", type=float, default=LEARNING_RATE,
                       help=f"Learning rate (default: {LEARNING_RATE})")
    
    # Hardware arguments
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                       help=f"Number of data loading workers (default: {NUM_WORKERS})")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto",
                       help="Device to train on (default: auto)")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()

def main():
    """Main training pipeline."""
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Determine device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
 
    print("=" * 60)
    print("FluxFish Grandmaster Training Pipeline")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data file: {args.data}")
    print(f"Model save path: {args.save}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Workers: {args.workers}")
    print(f"Device: {device}")
    print(f"Data file: {args.data}")
    print(f"Model save path: {args.save}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Workers: {args.workers}")
    
    # Step 1: Load existing training data only
    if not os.path.exists(args.data):
        print(f"ERROR: {args.data} not found!")
        print("Please ensure the data file exists before training.")
        return
    
    dataset = ChessDataset(args.data)
    print(f"Loaded {len(dataset)} positions from {args.data}")
    
    # Step 2: Train the model
    if len(dataset) < 1000:
        print("ERROR: Not enough training data!")
        return
    
    train_model(dataset, epochs=args.epochs, batch_size=args.batch_size, 
                learning_rate=args.lr, num_workers=args.workers, 
                device=device, save_path=args.save, verbose=args.verbose)
    
    print("\n✓ Training complete! Run export_model.py next.")

if __name__ == "__main__":
    main()
