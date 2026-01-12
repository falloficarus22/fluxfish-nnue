import chess
from evaluate import Evaluator
from mcts import MCTS
import time

def self_play():
    print("Initializing Engine Brain...")
    brain = Evaluator("fluxfish.nnue")
    searcher = MCTS(brain)
    
    board = chess.Board()
    
    print("\n--- SELF PLAY START ---")
    print(board)
    
    while not board.is_game_over():
        # Engine thinking
        print(f"\nMove {board.fullmove_number} ({'White' if board.turn == chess.WHITE else 'Black'}) thinking...")
        
        start_time = time.time()
        # Using 300 iterations for a balance of speed and strength
        best_move = searcher.search(board, iterations=300)
        elapsed = time.time() - start_time
        
        if best_move is None: break
        
        board.push(best_move)
        
        print(f"Engine plays: {best_move} (took {elapsed:.2f}s)")
        print(board)
        
        # Optional: Sleep to make it watchable
        # time.sleep(0.5)

    print("\nGame Over!")
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    self_play()
