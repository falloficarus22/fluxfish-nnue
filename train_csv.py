"""
FluxFish NNUE Trainer (Optimized for GPU & Robust Data)
Supports direct CSV loading, data sanitization, and RAM pre-loading.
FluxFish NNUE Trainer (Optimized for GPU & Robust Data)
Supports direct CSV loading, data sanitization, and RAM pre-loading.
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
            # low_memory=False fixes the DtypeWarning and is safer for mixed types
            df = pd.read_csv(csv_file, low_memory=False)
            # low_memory=False fixes the DtypeWarning and is safer for mixed types
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Identify columns
            cols = df.columns.tolist()
            # Try to find columns case-insensitive
            # Try to find columns case-insensitive
            fen_col = next((c for c in cols if 'fen' in c.lower()), None) or cols[0]
            eval_col = next((c for c in cols if 'sc' in c.lower() or 'ev' in c.lower()), None) or (cols[1] if len(cols) > 1 else None)

            print(f"Identified columns - FEN: '{fen_col}', Score: '{eval_col}'")
            
            # --- DATA SANITIZATION START ---
            
            # 1. Drop rows where essential data is missing
            initial_len = len(df)
            df = df.dropna(subset=[fen_col, eval_col])
            
            # 2. Force Score column to numeric, turning errors (like strings) into NaN
            df[eval_col] = pd.to_numeric(df[eval_col], errors='coerce')
            
            # 3. Drop rows that became NaN after forcing numeric (e.g. malformed strings)
            df = df.dropna(subset=[eval_col])
            
            # 4. Filter FENs: Simple check to ensure they are strings and reasonable length
            df = df[df[fen_col].apply(lambda x: isinstance(x, str) and len(x) > 10)]
            
            final_len = len(df)
            if final_len < initial_len:
                print(f"⚠️ Dropped {initial_len - final_len} bad rows (NaNs, strings in scores, etc).")
            
            # --- DATA SANITIZATION START ---
            
            # 1. Drop rows where essential data is missing
            initial_len = len(df)
            df = df.dropna(subset=[fen_col, eval_col])
            
            # 2. Force Score column to numeric, turning errors (like strings) into NaN
            df[eval_col] = pd.to_numeric(df[eval_col], errors='coerce')
            
            # 3. Drop rows that became NaN after forcing numeric (e.g. malformed strings)
            df = df.dropna(subset=[eval_col])
            
            # 4. Filter FENs: Simple check to ensure they are strings and reasonable length
            df = df[df[fen_col].apply(lambda x: isinstance(x, str) and len(x) > 10)]
            
            final_len = len(df)
            if final_len < initial_len:
                print(f"⚠️ Dropped {initial_len - final_len} bad rows (NaNs, strings in scores, etc).")
            
            self.fens = df[fen_col].values
            self.evals = df[eval_col].values.astype(np.float32) # Ensure strictly float32
            
            # Check for infinites
            if not np.isfinite(self.evals).all():
                print("⚠️ Found infinite values in scores! Clipping...")
                self.evals = np.clip(self.evals, -100000, 100000)
            
            print(f"Score Stats: Min={self.evals.min():.2f}, Max={self.evals.max():.2f}, Mean={self.evals.mean():.2f}")
            
            # Normalize CP to [-1, 1]
            self.evals = df[eval_col].values.astype(np.float32) # Ensure strictly float32
            
            # Check for infinites
            if not np.isfinite(self.evals).all():
                print("⚠️ Found infinite values in scores! Clipping...")
                self.evals = np.clip(self.evals, -100000, 100000)
            
            print(f"Score Stats: Min={self.evals.min():.2f}, Max={self.evals.max():.2f}, Mean={self.evals.mean():.2f}")
            
            # Normalize CP to [-1, 1]
            if np.abs(self.evals).max() > 1.5:
                print("Normalizing centipawn scores to [-1, 1] range...")
                self.evals = np.tanh(self.evals / 400.0)
            
            # --- DATA SANITIZATION END ---
            
            
            # --- DATA SANITIZATION END ---
            
            self.use_preload = preload
            
            if self.use_preload:
                print("Pre-loading features into RAM...")
                self.w_features, self.b_features, self.stm = self._preload_all()
                print("Pre-loading complete!")
                
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        self.model = FluxFishNNUE() 
        print(f"Total valid positions: {len(self.fens)}")

    def _preload_all(self):
        w_all = torch.zeros(len(self.fens), 768)
        b_all = torch.zeros(len(self.fens), 768)
        stm_all = torch.zeros(len(self.fens))
        
        print("Processing... (0%)", end='\r')
        count_bad = 0
        for i, fen in enumerate(self.fens):
            try:
                board = chess.Board(fen)
                w_all[i] = self.model.get_features(board, chess.WHITE)
                b_all[i] = self.model.get_features(board, chess.BLACK)
                stm_all[i] = 1.0 if board.turn == chess.WHITE else 0.0
            except:
                # If a FEN is still somehow invalid, we'll just have a zero-vector (neutral)
                # and log it.
                count_bad += 1
                # If a FEN is still somehow invalid, we'll just have a zero-vector (neutral)
                # and log it.
                count_bad += 1
            
            if i % 5000 == 0:
            if i % 5000 == 0:
                print(f"Processing... ({int(i/len(self.fens)*100)}%)", end='\r')
        
        if count_bad > 0:
            print(f"\n⚠️ Encountered {count_bad} invalid FENs during preload (inputs set to 0).")
            
        
        if count_bad > 0:
            print(f"\n⚠️ Encountered {count_bad} invalid FENs during preload (inputs set to 0).")
            
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
    
    # LOWER LEARNING RATE TO PREVENT EXPLOSION
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # USE SMOOTH L1 LOSS (Less sensitive to outliers than MSE)
    criterion = nn.SmoothL1Loss()
    
    print(f"Starting training for {args.epochs} epochs on {device}...")
    print(f"Params: Batch={args.batch_size}, LR={args.lr}, Loss=SmoothL1")
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        start_time = time.time()
        
        for i, (w, b, s, y) in enumerate(dataloader):
            w, b, s, y = w.to(device), b.to(device), s.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(w, b, s) 
            
            # DEBUG: Check for explosion in first batch
            if i == 0 and epoch == 0:
                print(f"DEBUG Batch 0: Targets Min={y.min():.3f}, Max={y.max():.3f}")
                print(f"DEBUG Batch 0: Preds   Min={pred.min():.3f}, Max={pred.max():.3f}")

            loss = criterion(pred, y)
             # Check for NaN loss instantly
            if torch.isnan(loss):
                print(f"❌ Batch {i} produced NaN loss!")
                print("This usually means learning rate is too high or data is bad.")
                return 

            loss.backward()
             # Gradient Clipping prevents exploding gradients (fix for simple NaN cases)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
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
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 1e-4 for stability)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save-path", type=str, default="fluxfish.nnue", help="Path to save the trained model")
    parser.add_argument("--workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--no-preload", action="store_true", help="Disable RAM pre-loading")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: dataset file '{args.csv_file}' not found.")
        sys.exit(1)
        
    train(args)
