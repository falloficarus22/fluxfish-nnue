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
from nnue_model import FluxFishNNUE, NUM_FEATURES

# ============ CONFIGURATION ============
STOCKFISH_PATH = "/usr/games/stockfish"  # Update if needed
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_POSITIONS = 500000  # Total training positions
EPOCHS = 4
NUM_WORKERS = max(1, cpu_count() - 1)
SAVE_PATH = "fluxfish.nnue"

# Stockfish settings (fast evaluation)
SF_DEPTH = 10  # Low depth for speed
SF_TIME_LIMIT = 0.01  # 10ms per position


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
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    self.positions.append((parts[0], float(parts[1])))
        print(f"Loaded {len(self.positions)} positions from {path}")
    
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
    moves_played = random.randint(4, depth)
    
    for _ in range(moves_played):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    
    return board


def evaluate_with_stockfish(fen: str) -> float:
    """Get Stockfish evaluation for a position."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        board = chess.Board(fen)
        
        # Quick analysis
        info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
        score = info["score"].relative
        
        engine.quit()
        
        # Convert to [-1, 1] range
        if score.is_mate():
            return 1.0 if score.mate() > 0 else -1.0
        else:
            # Clamp centipawns to reasonable range and normalize
            cp = max(-1000, min(1000, score.score()))
            return np.tanh(cp / 400.0)  # 400cp = ~0.76
            
    except Exception as e:
        return 0.0


def generate_position_batch(args) -> list:
    """Generate a batch of positions with evaluations (for multiprocessing)."""
    batch_size, worker_id = args
    results = []
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        
        for i in range(batch_size):
            board = generate_random_position(random.randint(6, 40))
            
            if board.is_game_over():
                continue
            
            # Get Stockfish evaluation
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
                score = info["score"].relative
                
                if score.is_mate():
                    ev = 1.0 if score.mate() > 0 else -1.0
                else:
                    cp = max(-1000, min(1000, score.score()))
                    ev = np.tanh(cp / 400.0)
                
                results.append((board.fen(), ev))
                
            except:
                continue
        
        engine.quit()
        
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
    
    return results


def generate_dataset_parallel(num_positions: int, output_file: str):
    """Generate training data using multiple processes."""
    print(f"Generating {num_positions} positions using {NUM_WORKERS} workers...")
    
    # Split work among workers
    positions_per_worker = num_positions // NUM_WORKERS
    work_items = [(positions_per_worker, i) for i in range(NUM_WORKERS)]
    
    dataset = ChessDataset()
    start_time = time.time()
    
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(generate_position_batch, work_items)
    
    for batch in results:
        for fen, ev in batch:
            dataset.add_position(fen, ev)
    
    elapsed = time.time() - start_time
    print(f"Generated {len(dataset)} positions in {elapsed:.1f}s")
    print(f"Rate: {len(dataset) / elapsed:.1f} positions/second")
    
    dataset.save(output_file)
    return dataset


def train_model(dataset: ChessDataset, epochs: int = EPOCHS):
    """Train the NNUE model on the dataset."""
    device = torch.device('cpu')
    model = FluxFishNNUE().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead on CPU
        pin_memory=False
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
            
            # Forward pass (handle batched stm)
            outputs = []
            for i in range(len(w_feat)):
                out = model(w_feat[i], b_feat[i], stm[i].item())
                outputs.append(out)
            outputs = torch.stack(outputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, SAVE_PATH)
            print(f"  Saved best model (loss: {avg_loss:.4f})")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    
    return model


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("FluxFish NNUE Training - CPU Optimized")
    print("=" * 60)
    
    dataset_file = "master_data.txt"
    
    # Step 1: Generate or load training data
    if os.path.exists(dataset_file):
        print(f"\nFound existing dataset: {dataset_file}")
        dataset = ChessDataset(dataset_file)
    else:
        print("\nGenerating training dataset...")
        dataset = generate_dataset_parallel(NUM_POSITIONS, dataset_file)
    
    # Step 2: Train the model
    if len(dataset) < 100:
        print("ERROR: Not enough training data!")
        return
    
    model = train_model(dataset, epochs=EPOCHS)
    
    # Step 3: Quick test
    print("\n" + "=" * 60)
    print("Testing trained model...")
    print("=" * 60)
    
    from nnue_model import FluxFishEvaluator
    evaluator = FluxFishEvaluator(SAVE_PATH)
    
    # Test positions
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"),
        chess.Board("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"),
    ]
    
    for board in test_positions:
        ev = evaluator.evaluate(board)
        print(f"Position: {board.fen()[:40]}...")
        print(f"  Evaluation: {ev:+.3f}")
    
    print("\n✓ Training complete! Run main.py to play against the engine.")


if __name__ == "__main__":
    main()
