#!/usr/bin/env python3
"""
Evaluate RL-trained model performance.
"""

import torch
import chess
import time
from nnue_model import FluxFishNNUE
from mcts import MCTS

def evaluate_model(model_path: str, num_positions: int = 10):
    """Test model on various positions."""
    print(f"Loading model from {model_path}...")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model = FluxFishNNUE()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    def evaluator(board):
        with torch.no_grad():
            w_feat = model.get_features(board, chess.WHITE)
            b_feat = model.get_features(board, chess.BLACK)
            stm = board.turn == chess.WHITE
            value = model(w_feat.unsqueeze(0), b_feat.unsqueeze(0), stm)
            return value.item()
    
    mcts = MCTS(evaluator)
    
    # Test positions
    test_positions = [
        chess.Board(),  # Start position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After e4
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",  # Complex
    ]
    
    print(f"\nTesting {len(test_positions)} positions:")
    
    for i, pos in enumerate(test_positions):
        if isinstance(pos, str):
            board = chess.Board(pos)
        else:
            board = pos
            
        print(f"\nPosition {i+1}:")
        print(board)
        
        start_time = time.time()
        ranked_moves = mcts.search_ranked(board, iterations=500)
        elapsed = time.time() - start_time
        
        if ranked_moves:
            best_move = ranked_moves[0][0]
            print(f"Best move: {best_move}")
            print(f"Top 3 moves: {[m.uci() for m, _ in ranked_moves[:3]]}")
            print(f"Search time: {elapsed:.2f}s")
        else:
            print("No moves found")

if __name__ == "__main__":
    evaluate_model("fluxfish_rl.nnue")
