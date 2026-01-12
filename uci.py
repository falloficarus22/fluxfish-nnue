#!/usr/bin/env python3
"""
FluxFish UCI Interface
Implements the Universal Chess Interface protocol for Lichess deployment.
"""

import sys
import chess
from nnue_model import FluxFishEvaluator
from mcts import MCTS

# Engine info
ENGINE_NAME = "FluxFish"
ENGINE_AUTHOR = "FluxFish Team"
MODEL_PATH = "fluxfish.nnue"

# Piece values for blunder detection
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
}


class FluxFishUCI:
    def __init__(self):
        self.board = chess.Board()
        self.evaluator = None
        self.searcher = None
        self.debug = False
        
    def load_engine(self):
        """Load the NNUE model."""
        try:
            self.evaluator = FluxFishEvaluator(MODEL_PATH)
            self.searcher = MCTS(self.evaluator.evaluate)
            return True
        except Exception as e:
            self.send(f"info string Error loading model: {e}")
            return False
    
    def send(self, msg):
        """Send a message to the GUI."""
        print(msg, flush=True)
    
    def get_material_balance(self, board, perspective):
        """Get material balance for blunder detection."""
        total = 0
        for pt in PIECE_VALUES:
            total += len(board.pieces(pt, perspective)) * PIECE_VALUES[pt]
            total -= len(board.pieces(pt, not perspective)) * PIECE_VALUES[pt]
        return total
    
    def is_blunder(self, board, move, threshold=150):
        """Check if move loses material."""
        us = board.turn
        mat_before = self.get_material_balance(board, us)
        
        board.push(move)
        worst = mat_before
        
        for resp in board.legal_moves:
            if board.is_capture(resp):
                board.push(resp)
                mat_after = self.get_material_balance(board, us)
                
                # Check recapture
                best_recap = mat_after
                for recap in board.legal_moves:
                    if board.is_capture(recap):
                        board.push(recap)
                        best_recap = max(best_recap, self.get_material_balance(board, us))
                        board.pop()
                
                worst = min(worst, best_recap)
                board.pop()
        
        board.pop()
        return (mat_before - worst) > threshold
    
    def find_best_move(self, time_limit=None):
        """Find the best move using MCTS with blunder filter."""
        # Determine iterations based on time
        if time_limit:
            # Rough estimate: 500 iterations takes ~2-3 seconds
            iterations = min(800, max(200, int(time_limit / 5)))
        else:
            iterations = 500
        
        ranked = self.searcher.search_ranked(self.board, iterations=iterations)
        
        if not ranked:
            return None
        
        # Apply blunder filter
        for move, _ in ranked:
            if not self.is_blunder(self.board, move):
                return move
        
        return ranked[0][0]
    
    def uci_loop(self):
        """Main UCI communication loop."""
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            
            if not line:
                continue
            
            tokens = line.split()
            cmd = tokens[0]
            
            if cmd == "uci":
                self.send(f"id name {ENGINE_NAME}")
                self.send(f"id author {ENGINE_AUTHOR}")
                self.send("option name Threads type spin default 1 min 1 max 1")
                self.send("uciok")
                
            elif cmd == "isready":
                if self.evaluator is None:
                    self.load_engine()
                self.send("readyok")
                
            elif cmd == "ucinewgame":
                self.board = chess.Board()
                
            elif cmd == "position":
                self.parse_position(tokens[1:])
                
            elif cmd == "go":
                move = self.parse_go(tokens[1:])
                if move:
                    self.send(f"bestmove {move.uci()}")
                else:
                    # No legal moves
                    self.send("bestmove 0000")
                    
            elif cmd == "quit":
                break
                
            elif cmd == "debug":
                self.debug = tokens[1] == "on" if len(tokens) > 1 else False
                
            elif cmd == "stop":
                pass  # We don't support pondering, so nothing to stop
    
    def parse_position(self, tokens):
        """Parse the position command."""
        idx = 0
        
        if tokens[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
            fen_parts = []
            idx += 1
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            self.board = chess.Board(" ".join(fen_parts))
        
        # Apply moves
        if idx < len(tokens) and tokens[idx] == "moves":
            idx += 1
            while idx < len(tokens):
                move = chess.Move.from_uci(tokens[idx])
                self.board.push(move)
                idx += 1
    
    def parse_go(self, tokens):
        """Parse the go command and return the best move."""
        time_limit = None
        
        # Parse time controls
        i = 0
        while i < len(tokens):
            if tokens[i] == "wtime" and self.board.turn == chess.WHITE:
                time_limit = int(tokens[i + 1]) / 1000  # Convert to seconds
                i += 2
            elif tokens[i] == "btime" and self.board.turn == chess.BLACK:
                time_limit = int(tokens[i + 1]) / 1000
                i += 2
            elif tokens[i] == "movetime":
                time_limit = int(tokens[i + 1]) / 1000
                i += 2
            elif tokens[i] == "infinite":
                time_limit = 30  # Default to 30 seconds for infinite
                i += 1
            else:
                i += 1
        
        # Default time limit
        if time_limit is None:
            time_limit = 5
        
        # Use a fraction of remaining time
        think_time = min(time_limit / 30, 10)  # At most 10 seconds
        
        if self.debug:
            self.send(f"info string thinking for {think_time:.1f}s")
        
        return self.find_best_move(think_time * 1000)


def main():
    engine = FluxFishUCI()
    engine.uci_loop()


if __name__ == "__main__":
    main()
