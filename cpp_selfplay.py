#!/usr/bin/env python3
"""
C++ Backend Self-Play Integration
Uses fluxfish_cpp engine for self-play games and extracts training data.
"""

import subprocess
import chess
import numpy as np
import time
from typing import List, Dict, Optional
import logging

class CppSelfPlay:
    """Self-play using C++ fluxfish engine."""
    
    def __init__(self, engine_path: str = "./cpp/fluxfish_cpp"):
        self.engine_path = engine_path
        self.logger = logging.getLogger(__name__)
        
    def play_game(self, max_moves: int = 200, time_per_move: int = 100) -> List[Dict]:
        """Play a complete self-play game and extract training data."""
        
        # Start C++ engine
        process = subprocess.Popen(
            ["wsl", "-d", "Ubuntu", self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        experiences = []
        board = chess.Board()
        move_count = 0
        
        try:
            # Initialize engine
            self._send_command(process, "uci")
            self._send_command(process, "ucinewgame")
            
            while not board.is_game_over() and move_count < max_moves:
                # Get position features before move
                features = self._extract_features(board)
                
                # Ask engine for best move
                self._send_command(process, f"position fen {board.fen()}")
                self._send_command(process, f"go movetime {time_per_move}")
                
                # Parse engine response
                best_move, engine_info = self._parse_engine_response(process)
                
                if best_move is None:
                    break
                
                # Extract policy from engine info (simplified)
                policy = self._extract_policy_from_info(engine_info, board)
                
                # Store experience
                experiences.append({
                    'features': features,
                    'policy': policy,
                    'board_fen': board.fen(),
                    'move': best_move.uci(),
                    'move_count': move_count
                })
                
                # Make move
                board.push(best_move)
                move_count += 1
            
            # Get game result
            result = self._get_game_result(board)
            
            # Update all experiences with result
            for exp in experiences:
                exp['result'] = result
            
            return experiences
            
        except Exception as e:
            self.logger.error(f"Error during self-play: {e}")
            return []
        finally:
            try:
                self._send_command(process, "quit")
                process.wait(timeout=5)
            except:
                pass
    
    def _send_command(self, process, command):
        """Send command to engine."""
        process.stdin.write(command + "\n")
        process.stdin.flush()
    
    def _parse_engine_response(self, process, timeout: float = 5.0) -> tuple:
        """Parse engine response to get best move and info."""
        best_move = None
        info_lines = []
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = process.stdout.readline()
            if not line:
                break
            
            line = line.strip()
            info_lines.append(line)
            
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        best_move = chess.Move.from_uci(parts[1])
                    except:
                        pass
                break
        
        return best_move, info_lines
    
    def _extract_features(self, board: chess.Board) -> np.ndarray:
        """Extract NNUE features from board position."""
        # This should match your NNUE model's feature extraction
        features = np.zeros(41024, dtype=np.float32)  # Adjust size based on your model
        
        # Simplified feature extraction - implement proper NNUE features
        square_index = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Convert piece to feature index
                piece_type = piece.piece_type - 1  # 0-5
                color = 0 if piece.color == chess.WHITE else 1  # 0-1
                
                # This is simplified - implement proper NNUE feature mapping
                feature_idx = (piece_type * 2 + color) * 64 + square_index
                if feature_idx < len(features):
                    features[feature_idx] = 1.0
            
            square_index += 1
        
        return features
    
    def _extract_policy_from_info(self, info_lines: List[str], board: chess.Board) -> np.ndarray:
        """Extract policy from engine info lines."""
        # Create policy vector for all possible moves
        policy = np.zeros(4096, dtype=np.float32)  # Adjust size based on move encoding
        
        # This is simplified - proper implementation would parse engine search info
        # For now, create uniform policy over legal moves
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            # Convert move to policy index (simplified)
            move_idx = move.from_square() * 64 + move.to_square()
            if move_idx < len(policy):
                policy[move_idx] = 1.0 / len(legal_moves)
        
        return policy
    
    def _get_game_result(self, board: chess.Board) -> float:
        """Get game result from perspective of the player who just moved."""
        if board.is_checkmate():
            # The player who delivered checkmate wins
            return 1.0 if not board.turn else -1.0
        elif board.is_stalemate() or board.can_claim_draw():
            return 0.0
        else:
            return 0.0  # Default to draw

def test_selfplay():
    """Test the self-play system."""
    print("Testing C++ self-play...")
    
    selfplay = CppSelfPlay()
    
    # Play a test game
    experiences = selfplay.play_game(max_moves=20, time_per_move=50)
    
    print(f"Played game with {len(experiences)} positions")
    if experiences:
        print(f"First position: {experiences[0]['board_fen']}")
        print(f"Result: {experiences[0]['result']}")
        print("✅ Self-play test successful!")
    else:
        print("❌ Self-play test failed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_selfplay()
