import chess
import chess.engine
import chess.pgn
import os
import random
import numpy as np
import time
import argparse
from multiprocessing import Pool, cpu_count

# ============ DEFAULT CONFIGURATION ============
DEFAULT_STOCKFISH = "/usr/games/stockfish"
DEFAULT_OUTPUT = "master_data.txt"
DEFAULT_POSITIONS = 5000000  # 5 Million for "World Class" attempt
DEFAULT_DEPTH = 14           # Depth 14-16 is the "Sweet Spot" for high quality without being too slow
# ===============================================

def evaluate_position(args_tuple):
    """Analysis function used by workers."""
    engine_path, fen, depth = args_tuple
    try:
        # We need a fresh engine instance per process usually, 
        # but creating it for every single position is slow.
        # This helper is called inside the worker loop where engine exists.
        # So we refactor worker_task to handle engine life-cycle.
        pass 
    except:
        return None

# Helper to normalize score
def normalize_score(score):
    if score.is_mate():
        return 1.0 if score.mate() > 0 else -1.0
    else:
        # Clamp to range [-2000, 2000] for normalization
        cp = max(-2000, min(2000, score.score()))
        # Scale: 300cp is big advantage. tanh(300/400) = ~0.6
        return np.tanh(cp / 400.0)

def worker_task(args):
    """Generates high-quality positions by playing out games."""
    num_to_gen, worker_id, engine_path, depth = args
    results = []
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        
        while len(results) < num_to_gen:
            board = chess.Board()
            # Play a semi-random game
            while not board.is_game_over() and len(board.move_stack) < 150:
                # 20% chance of random move to diversify openings
                if random.random() < 0.2:
                    if len(list(board.legal_moves)) > 0:
                        move = random.choice(list(board.legal_moves))
                    else:
                        break
                else:
                    # Quick best move to keep game realistic
                    try:
                        # Use lower depth for game generation to move fast
                        play_depth = max(1, depth // 2)
                        info = engine.analyse(board, chess.engine.Limit(depth=play_depth)) 
                        move = info.get("pv", [None])[0]
                        if not move: 
                            move = random.choice(list(board.legal_moves))
                    except:
                         if len(list(board.legal_moves)) > 0:
                            move = random.choice(list(board.legal_moves))
                         else:
                            break
                
                board.push(move)
                
                # Sample 10% of positions from the game (after move 8)
                if random.random() < 0.1 and len(board.move_stack) > 8:
                    try:
                        # HIGH QUALITY EVALUATION
                        info = engine.analyse(board, chess.engine.Limit(depth=depth))
                        score = info["score"].relative
                        ev = normalize_score(score)
                        
                        results.append(f"{board.fen()}|{ev}")
                        if len(results) >= num_to_gen: break
                    except:
                        pass
                        
            if worker_id == 0 and len(results) % 100 == 0:
                 print(f"Worker 0: {len(results)}/{num_to_gen}", end='\r')

        engine.quit()
    except Exception as e:
        print(f"Worker {worker_id} Error: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="FluxFish Data Generator (World Class Edition)")
    
    parser.add_argument("--positions", type=int, default=DEFAULT_POSITIONS, help="Target number of positions (default: 5M)")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Stockfish analysis depth (default: 14)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output file path")
    parser.add_argument("--engine", type=str, default=DEFAULT_STOCKFISH, help="Path to Stockfish binary")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count()), help="Number of CPU workers")
    
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"FluxFish Data Generator")
    print(f"=" * 60)
    print(f"Goal:   {args.positions} positions")
    print(f"Depth:  {args.depth} (Teacher Quality)")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output}")
    
    # Check engine
    if not os.path.exists(args.engine) and "stockfish" not in args.engine: 
        # simplistic check, assume "stockfish" might be in path
        print(f"⚠️ Warning: Engine path '{args.engine}' might be invalid.")
    
    # 1. Check existing data to resume
    existing_count = 0
    if os.path.exists(args.output):
        print(f"Counting existing lines in {args.output}...")
        try:
             with open(args.output, 'r') as f:
                for _ in f: existing_count += 1
        except:
             pass
        print(f"Found {existing_count} existing positions.")
    
    remaining = args.positions - existing_count
    if remaining <= 0:
        print("Dataset is already complete!")
        return

    print(f"Generating {remaining} new positions...")
    
    # 2. Chunked Generation (Save periodically)
    CHUNK_SIZE = 5000 
    total_generated = 0
    
    start_time = time.time()
    
    while total_generated < remaining:
        current_batch_size = min(CHUNK_SIZE * args.workers, remaining - total_generated)
        per_worker = current_batch_size // args.workers
        
        if per_worker == 0: break
        
        # Pass args to worker
        worker_args = [(per_worker, i, args.engine, args.depth) for i in range(args.workers)]
        
        with Pool(args.workers) as pool:
            all_batches = pool.map(worker_task, worker_args)
        
        # Write to file
        with open(args.output, "a") as f:
            for batch in all_batches:
                for line in batch:
                    f.write(line + "\n")
                    total_generated += 1
        
        elapsed = time.time() - start_time
        rate = total_generated / elapsed if elapsed > 0 else 0
        total_now = existing_count + total_generated
        
        print(f"Progress: {total_now}/{args.positions} ({int(total_now/args.positions*100)}%) | Rate: {rate:.1f} pos/s", end='\r')

    print(f"\n\nDone! Data saved to {args.output}")

if __name__ == "__main__":
    main()
