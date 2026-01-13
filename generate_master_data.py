import chess
import chess.engine
import chess.pgn
import os
import random
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from nnue_model import FluxFishNNUE

# ============ CONFIGURATION ============
STOCKFISH_PATH = "/usr/games/stockfish"
OUTPUT_FILE = "master_data_100k.txt"
NUM_POSITIONS = 500000 # Targeted number of high-quality positions
SF_DEPTH = 12
NUM_WORKERS = max(1, cpu_count() - 1)

def evaluate_position(engine, board):
    """Analysis at depth 12 is much more reliable than depth 10."""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
        score = info["score"].relative
        
        if score.is_mate():
            ev = 1.0 if score.mate() > 0 else -1.0
        else:
            cp = max(-1000, min(1000, score.score()))
            ev = np.tanh(cp / 400.0)
        return ev
    except:
        return None

def worker_task(num_to_gen):
    """Generates high-quality positions by playing out games with moderate randomness."""
    results = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        while len(results) < num_to_gen:
            board = chess.Board()
            # Play a semi-random game
            while not board.is_game_over() and len(board.move_stack) < 100:
                # 20% chance of random move to diversify, otherwise best move
                if random.random() < 0.2:
                    move = random.choice(list(board.legal_moves))
                else:
                    info = engine.analyse(board, chess.engine.Limit(depth=6)) # Quick best move
                    move = info.get("pv", [None])[0] or random.choice(list(board.legal_moves))
                
                board.push(move)
                
                # Sample 10% of positions from the game
                if random.random() < 0.1 and len(board.move_stack) > 10:
                    ev = evaluate_position(engine, board)
                    if ev is not None:
                        results.append(f"{board.fen()}|{ev}")
                        if len(results) >= num_to_gen: break
            
        engine.quit()
    except Exception as e:
        print(f"Worker Error: {e}")
    return results

def main():
    print(f"Starting High-Quality Data Generation...")
    print(f"Goal: {NUM_POSITIONS} positions using {NUM_WORKERS} workers at depth {SF_DEPTH}.")
    
    start_time = time.time()
    per_worker = NUM_POSITIONS // NUM_WORKERS
    
    with Pool(NUM_WORKERS) as pool:
        all_batches = pool.map(worker_task, [per_worker] * NUM_WORKERS)
    
    with open(OUTPUT_FILE, "w") as f:
        count = 0
        for batch in all_batches:
            for line in batch:
                f.write(line + "\n")
                count += 1
                
    elapsed = time.time() - start_time
    print(f"Generated {count} positions in {elapsed:.1f}s ({count/elapsed:.1f} pos/s)")
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
