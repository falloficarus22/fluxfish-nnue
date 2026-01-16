"""
FluxFish NNUE Trainer (Optimized for GPU)
Supports direct CSV loading and RAM pre-loading for fast training.
"""
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
import time
from nnue_model import FluxFishNNUE

class CSVChessDataset(Dataset):
    """Dataset of playing positions from a CSV file."""
    
    def __init__(self, csv_file, preload=True):
        print(f"Loading data from {csv_file}...")
        try:
            df = pd.read_csv(csv_file)
            
            # Identify columns
            cols = df.columns.tolist()
            fen_col = next((c for c in cols if 'fen' in c.lower()), None) or cols[0]
            eval_col = next((c for c in cols if 'sc' in c.lower() or 'ev' in c.lower()), None) or (cols[1] if len(cols) > 1 else None)

            print(f"Identified columns - FEN: '{fen_col}', Score: '{eval_col}'")
            
            self.fens = df[fen_col].values
            self.evals = df[eval_col].values if eval_col else np.zeros(len(df))
            
            if np.abs(self.evals).max() > 1.5:
                print("Normalizing centipawn scores to [-1, 1] range...")
                self.evals = np.tanh(self.evals / 400.0)
                
            self.use_preload = preload
            
            if self.use_preload:
                print("Pre-loading features into RAM for maximum GPU speed... (This takes a few minutes but makes training instant)")
                self.w_features, self.b_features, self.stm = self._preload_all()
                print("Pre-loading complete!")
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)
        
        self.model = FluxFishNNUE() 
        print(f"Total positions: {len(self.fens)}")

    def _preload_all(self):
        # Robust Sequential Preload to avoid multiprocessing issues in simple scripts
        w_all = torch.zeros(len(self.fens), 768)
        b_all = torch.zeros(len(self.fens), 768)
        stm_all = torch.zeros(len(self.fens))
        
        print("Processing... (0%)", end='\r')
        for i, fen in enumerate(self.fens):
            try:
                board = chess.Board(fen)
                w_all[i] = self.model.get_features(board, chess.WHITE)
                b_all[i] = self.model.get_features(board, chess.BLACK)
                stm_all[i] = 1.0 if board.turn == chess.WHITE else 0.0
            except:
                pass
            
            if i % 1000 == 0:
                print(f"Processing... ({int(i/len(self.fens)*100)}%)", end='\r')
                
        return w_all, b_all, stm_all

    def __len__(self):
        return len(self.fens)
    
    def __getitem__(self, idx):
        if self.use_preload:
            # Instant access
            return self.w_features[idx], self.b_features[idx], self.stm[idx], torch.tensor([self.evals[idx]], dtype=torch.float32)
        else:
            # Slow fallback
            fen = self.fens[idx]
            board = chess.Board(fen)
            w = self.model.get_features(board, chess.WHITE).float()
            b = self.model.get_features(board, chess.BLACK).float()
            stm = torch.tensor([1.0 if board.turn == chess.WHITE else 0.0], dtype=torch.float32)
            label = torch.tensor([self.evals[idx]], dtype=torch.float32)
            return w, b, stm, label

def train(args):
    # FORCE CUDA CHECK
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("⚠️ WARNING: GPU NOT DETECTED! Training will be slow.")
        device = torch.device('cpu')
    
    # Enable preload by default unless --no-preload is set
    dataset = CSVChessDataset(args.csv_file, preload=not args.no_preload)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, # Workers not needed if preloaded; improves stability
        pin_memory=True
    )
    
    model = FluxFishNNUE().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    print(f"Starting training for {args.epochs} epochs on {device}...")
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        start_time = time.time()
        
        for i, (w, b, s, y) in enumerate(dataloader):
            w, b, s, y = w.to(device), b.to(device), s.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(w, b, s) 
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.5f} | Time: {elapsed:.1f}s")
        
        # Save checkpoint
        torch.save(model.state_dict(), args.save_path)
        
    print(f"Training complete. Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FluxFish NNUE from CSV data")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing training data (FEN, score)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save-path", type=str, default="fluxfish.nnue", help="Path to save the trained model")
    parser.add_argument("--workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--no-preload", action="store_true", help="Disable RAM pre-loading (use for huge datasets that don't fit in RAM)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: dataset file '{args.csv_file}' not found.")
        sys.exit(1)
        
    train(args)
