import chess
import numpy as np
from evaluate import Evaluator

def debug_engine():
    brain = Evaluator("nn-1c0000000000.nnue")
    
    # Test 1: Starting Position (Should be near 0.0 or slightly positive)
    board = chess.Board()
    start_eval = brain.evaluate(board)
    print(f"Start Position Eval: {start_eval:.4f}")

    # Test 2: Blunder Test (White loses Queen)
    board.remove_piece_at(chess.D1)
    blunder_eval = brain.evaluate(board)
    print(f"White loses Queen Eval: {blunder_eval:.4f}")

    if abs(start_eval - blunder_eval) < 0.01:
        print("CRITICAL: The engine can't tell the difference between having a Queen and not.")
        print("This means the weights are not loading correctly or the math is wrong.")
    else:
        print("SUCCESS: The engine recognizes material loss.")

if __name__ == "__main__":
    debug_engine()