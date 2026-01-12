"""
FluxFish Chess Engine - Main game loop.
Uses our custom-trained NNUE model with MCTS search.
Includes blunder filter to prevent obvious material losses.
"""

import chess
import os
from nnue_model import FluxFishEvaluator
from mcts import MCTS

MODEL_PATH = "fluxfish.nnue"

# Piece values for material counting
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}


def count_material(board: chess.Board, color: chess.Color) -> int:
    """Count total material for a given color."""
    total = 0
    for piece_type in PIECE_VALUES:
        total += len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
    return total


def get_material_balance(board: chess.Board, perspective: chess.Color) -> int:
    """Get material balance from perspective's point of view."""
    return count_material(board, perspective) - count_material(board, not perspective)


def is_blunder(board: chess.Board, move: chess.Move, threshold: int = 150) -> bool:
    """
    Check if a move is a blunder by looking at immediate material loss.
    
    A move is a blunder if:
    1. After we play the move, opponent can capture something
    2. The resulting material swing is worse than -threshold centipawns
    """
    us = board.turn
    
    # Material before move
    material_before = get_material_balance(board, us)
    
    # Play our move
    board.push(move)
    
    # Check if we're in check (might be necessary sacrifice)
    gives_check = board.is_check()
    
    # Find the worst-case opponent response (best capture)
    worst_material = material_before
    
    for response in board.legal_moves:
        if board.is_capture(response):
            board.push(response)
            material_after = get_material_balance(board, us)
            
            # Check if we can recapture
            best_recapture = material_after
            for recapture in board.legal_moves:
                if board.is_capture(recapture):
                    board.push(recapture)
                    recapture_material = get_material_balance(board, us)
                    best_recapture = max(best_recapture, recapture_material)
                    board.pop()
            
            worst_material = min(worst_material, best_recapture)
            board.pop()
    
    board.pop()  # Undo our move
    
    # If we give check, be more lenient
    if gives_check:
        threshold += 200
    
    # It's a blunder if material drops significantly
    return (material_before - worst_material) > threshold


def filter_blunders(board: chess.Board, moves_with_scores: list) -> chess.Move:
    """
    Filter out blundering moves and return the best safe move.
    
    Args:
        board: Current board position
        moves_with_scores: List of (move, score) tuples sorted by score descending
    
    Returns:
        Best non-blundering move, or best move if all blunder
    """
    for move, score in moves_with_scores:
        if not is_blunder(board, move):
            return move
    
    # If all moves blunder, return the one with highest MCTS score
    return moves_with_scores[0][0] if moves_with_scores else None

def main():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("=" * 50)
        print("ERROR: No trained model found!")
        print("Please run 'python train_fast.py' first to train the NNUE.")
        print("=" * 50)
        return
    
    # Load our custom NNUE
    print("Loading FluxFish NNUE...")
    evaluator = FluxFishEvaluator(MODEL_PATH)
    searcher = MCTS(evaluator.evaluate)
    
    board = chess.Board()
    
    print("\n" + "=" * 50)
    print("FluxFish Chess Engine")
    print("You play as WHITE, engine plays as BLACK")
    print("Enter moves in UCI format (e.g., e2e4, g1f3)")
    print("=" * 50 + "\n")

    while not board.is_game_over():
        print(board)
        print()

        if board.turn == chess.WHITE:
            move = input("Your Move (UCI): ").strip()
            
            if move.lower() == 'quit':
                print("Thanks for playing!")
                break

            try:
                board.push_uci(move)
            except ValueError:
                print("Invalid move! Try again.")
                continue

        else:
            print("FluxFish is thinking...")
            ranked_moves = searcher.search_ranked(board, iterations=500)
            
            if not ranked_moves:
                print("Engine has no legal moves!")
                break
            
            # Apply blunder filter
            best_move = filter_blunders(board, ranked_moves)
            
            if best_move is None:
                best_move = ranked_moves[0][0]
            
            # Show if we avoided a blunder
            original_best = ranked_moves[0][0]
            if best_move != original_best:
                print(f"  (Avoided blunder: {original_best})")
                
            print(f"FluxFish plays: {best_move}")
            board.push(best_move)
        
        print()

    # Game over
    print("\n" + "=" * 50)
    print("Game Over!")
    print(f"Result: {board.result()}")
    print("=" * 50)


if __name__ == "__main__":
    main()