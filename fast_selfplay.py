#!/usr/bin/env python3
"""
Fast C++-based self-play for RL training.
Uses the compiled C++ engine for rapid game generation.
"""

import chess
import subprocess
import tempfile
import os
import time
import random
from typing import List, Tuple

class FastSelfPlay:
    """C++ accelerated self-play for rapid game generation."""
    
    def __init__(self, cpp_engine_path: str = "cpp/fluxfish_cpp"):
        self.cpp_engine_path = cpp_engine_path
        self.temperature_threshold = 30  # Move count before temperature drops
        
    def get_cpp_move(self, fen: str, iterations: int = 200) -> chess.Move:
        """Get move from C++ engine."""
        try:
            # Create temporary FEN file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fen', delete=False) as f:
                f.write(fen)
                temp_path = f.name
            
            # Call C++ engine
            cmd = [self.cpp_engine_path, temp_path, str(iterations)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Clean up
            os.unlink(temp_path)
            
            if result.returncode == 0:
                # Parse output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "Best move found:" in line:
                        move_str = line.split(":")[1].strip()
                        return chess.Move.from_uci(move_str)
            
            # Fallback to random move
            board = chess.Board(fen)
            return random.choice(list(board.legal_moves))
            
        except Exception as e:
            print(f"C++ engine error: {e}")
            # Fallback to random move
            board = chess.Board(fen)
            return random.choice(list(board.legal_moves))
    
    def get_move_probabilities(self, board: chess.Board, iterations: int = 200) -> List[float]:
        """Get probability distribution using C++ engine with exploration."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []
        
        # Get best move from C++ engine
        best_move = self.get_cpp_move(board.fen(), iterations)
        
        # Create probability distribution with more exploration
        move_probs = []
        for move in legal_moves:
            if move == best_move:
                move_probs.append(0.5)  # Reduced probability for best move
            else:
                move_probs.append(0.5 / (len(legal_moves) - 1))  # Distribute remaining probability
        
        return move_probs
    
    def play_fast_game(self, max_moves: int = 200, mcts_iterations: int = 200) -> List[Tuple[str, float, List[float]]]:
        """Play a game using C++ engine for fast move generation."""
        board = chess.Board()
        game_data = []
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            # Get move probabilities
            move_probs = self.get_move_probabilities(board, mcts_iterations)
            
            # Sample move according to probabilities
            move_idx = random.choices(range(len(legal_moves)), weights=move_probs)[0]
            move = legal_moves[move_idx]
            
            # Store position data
            fen = board.fen()
            # Use simple material-based evaluation for speed
            current_value = self.simple_material_eval(board)
            game_data.append((fen, current_value, move_probs))
            
            # Make move
            board.push(move)
            move_count += 1
        
        # Assign final values based on game result
        result = board.result()
        final_value = 0.0
        if result == "1-0":
            final_value = 1.0
        elif result == "0-1":
            final_value = -1.0
        
        # Update game data with final outcome
        for i, (fen, _, policy) in enumerate(game_data):
            # Weight positions towards the end of the game
            weight = (i + 1) / len(game_data)
            value = final_value * weight + game_data[i][1] * (1 - weight)
            game_data[i] = (fen, value, policy)
        
        return game_data
    
    def simple_material_eval(self, board: chess.Board) -> float:
        """Fast material-based evaluation for game data."""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        score = 0
        for piece_type in piece_values:
            white_pieces = len(board.pieces(piece_type, chess.WHITE))
            black_pieces = len(board.pieces(piece_type, chess.BLACK))
            score += (white_pieces - black_pieces) * piece_values[piece_type]
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, score / 20.0))

def benchmark_self_play():
    """Benchmark Python vs C++ self-play speed."""
    print("=== Self-Play Speed Benchmark ===")
    
    # Test C++ speed
    fast_play = FastSelfPlay()
    
    print("Testing C++ self-play...")
    start_time = time.time()
    game_data = fast_play.play_fast_game(mcts_iterations=100)
    cpp_time = time.time() - start_time
    
    print(f"C++ Self-Play: {cpp_time:.2f}s for {len(game_data)} positions")
    print(f"Speed: {len(game_data)/cpp_time:.1f} positions/second")
    
    # Test a few moves
    print(f"\nSample positions from game:")
    for i, (fen, value, _) in enumerate(game_data[:3]):
        board = chess.Board(fen)
        print(f"Move {i+1}: {board.san(board.peek()) if board.move_stack else 'Start'}")
        print(f"  Evaluation: {value:.3f}")

if __name__ == "__main__":
    benchmark_self_play()
