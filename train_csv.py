import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import chess
import numpy as np
import os
import sys
import argparse
from nnue_model import FluxFishNNUE

class CSVChessDataset(Dataset):
    """Dataset of playing positions from a CSV file."""
    
    def __init__(self, csv_file):
        print(f"Loading data from {csv_file}...")
        try:
            # Try to read with pandas first for speed
            # Assumes columns like 'FEN', 'Evaluation' or just no header: FEN, Score
            # We'll try to sniff the format
            df = pd.read_csv(csv_file)
            
            # Identify columns
            cols = df.columns.tolist()
            fen_col = next((c for c in cols if 'fen' in c.lower()), None)
            eval_col = next((c for c in cols if 'sc' in c.lower() or 'ev' in c.lower()), None)
            
            if not fen_col:
                # Fallback: assume first column is FEN
                fen_col = cols[0]
            if not eval_col:
                # Fallback: assume second column is Score
                eval_col = cols[1] if len(cols) > 1 else None

            print(f"Identified columns - FEN: '{fen_col}', Score: '{eval_col}'")
            
            self.fens = df[fen_col].values
            self.evals = df[eval_col].values if eval_col else np.zeros(len(df))
            
            # Normalize scores if they look like centipawns (large numbers)
            # Typically SF scores are in centipawns. We need [-1, 1] for training usually,
            # or we train to predict raw CP. NNUE usually trains on [0, 1] or sigmoid-like.
            # Stockfish NNUE actually trains on a specific range.
            # Let's assume CP and normalize to sigmoid.
            if np.abs(self.evals).max() > 1.5:
                print("Normalizing centipawn scores to [-1, 1] range...")
                self.evals = np.tanh(self.evals / 400.0)
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)
            
        self.model = FluxFishNNUE() # For feature lookups
        print(f"Loaded {len(self.fens)} positions.")

    def __len__(self):
        return len(self.fens)
    
    def __getitem__(self, idx):
        fen = self.fens[idx]
        eval_score = self.evals[idx]
        
        board = chess.Board(fen)
        
        # Get features (this is slow in Python loop, but functional)
        w_feat = self.model.get_features(board, chess.WHITE).float()
        b_feat = self.model.get_features(board, chess.BLACK).float()
        
        # Side to move
        stm = torch.tensor([1.0 if board.turn == chess.WHITE else 0.0], dtype=torch.float32)
        label = torch.tensor([eval_score], dtype=torch.float32)
        
        return w_feat, b_feat, stm, label

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = CSVChessDataset(args.csv_file)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers
    )
    
    model = FluxFishNNUE().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Batch Size: {args.batch_size}, LR: {args.lr}")
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for i, (w, b, s, y) in enumerate(dataloader):
            w, b, s, y = w.to(device), b.to(device), s.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Model forward
            pred = model(w, b, s) 
            
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), args.save_path)
        print(f"Checkpoint saved to {args.save_path}")
        
    print(f"Training complete. Final model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FluxFish NNUE from CSV data")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing training data (FEN, score)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save-path", type=str, default="fluxfish.nnue", help="Path to save the trained model")
    parser.add_argument("--workers", type=int, default=0, help="Number of data loader workers (0 = main thread)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: dataset file '{args.csv_file}' not found.")
        sys.exit(1)
        
    train(args)
