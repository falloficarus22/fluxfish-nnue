#!/usr/bin/env python3
"""
FluxFish UCI Interface
<<<<<<< HEAD
Implements the Universal Chess Interface protocol for Lichess deployment.
"""

import sys
import chess
=======
Optimized for tactical strength and stable deployment.
"""

import sys
import os
import time
import chess
import chess.polyglot
>>>>>>> 0548ecf (First successful game played against the bot)
from nnue_model import FluxFishEvaluator
from mcts import MCTS

# Engine info
ENGINE_NAME = "FluxFish"
ENGINE_AUTHOR = "FluxFish Team"
MODEL_PATH = "fluxfish.nnue"
<<<<<<< HEAD

# Piece values for blunder detection
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
}


=======
BOOK_PATH = "komodo.bin"

# Piece values for tactical filtering
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

>>>>>>> 0548ecf (First successful game played against the bot)
class FluxFishUCI:
    def __init__(self):
        self.board = chess.Board()
        self.evaluator = None
        self.searcher = None
        self.debug = False
<<<<<<< HEAD
        
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
=======
        self.book = None
        
    def load_engine(self):
        """Load the NNUE model and opening book."""
        try:
            self.evaluator = FluxFishEvaluator(MODEL_PATH)
            self.searcher = MCTS(self.evaluator.evaluate)
            
            if os.path.exists(BOOK_PATH):
                self.book = BOOK_PATH
                self.send(f"info string Loaded opening book: {BOOK_PATH}")
                
            return True
        except Exception as e:
            self.send(f"info string Error loading engine: {e}")
            return False
    
    def send(self, msg):
        """Send a message to the UCI GUI."""
        print(msg, flush=True)

    def get_material_balance(self, board, perspective):
        """Calculate material balance from perspective."""
        total = 0
        for pt, val in PIECE_VALUES.items():
            total += len(board.pieces(pt, perspective)) * val
            total -= len(board.pieces(pt, not perspective)) * val
        return total

    def is_blunder_tactical(self, board, move):
        """A robust 2-ply tactical check for blunders."""
        us = board.turn
        mat_before = self.get_material_balance(board, us)
        
        # 1. Play our move
        board.push(move)
        
        # 1.1 Checkmate check
        if board.is_checkmate(): # We got checkmated!
            board.pop()
            return True
        
        tactical_error = False
        
        # 2. Look at opponent's best response (captures/checks)
        for resp in board.legal_moves:
            if board.is_capture(resp) or board.gives_check(resp):
                board.push(resp)
                
                # Opponent delivers mate
                if board.is_checkmate():
                    tactical_error = True
                    board.pop()
                    break
                
                # Opponent wins material (check recaptures)
                mat_after = self.get_material_balance(board, us)
>>>>>>> 0548ecf (First successful game played against the bot)
                best_recap = mat_after
                for recap in board.legal_moves:
                    if board.is_capture(recap):
                        board.push(recap)
                        best_recap = max(best_recap, self.get_material_balance(board, us))
                        board.pop()
                
<<<<<<< HEAD
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
=======
                # If we lose more than 1.5 pawns worth of material
                if (mat_before - best_recap) > 150:
                    tactical_error = True
                    board.pop()
                    break
                    
                board.pop()
        
        board.pop()
        return tactical_error

    def find_best_move(self, time_limit_s=None):
        """Find the best move using MCTS and tactical filtering."""
        # Use more iterations now that MCTS is faster
        max_its = 3000
        
        # Perform search
        # We give MCTS a dedicated time limit to ensure it returns
        ranked = self.searcher.search_ranked(self.board, iterations=max_its, time_limit=time_limit_s)
>>>>>>> 0548ecf (First successful game played against the bot)
        
        if not ranked:
            return None
        
<<<<<<< HEAD
        # Apply blunder filter
        for move, _ in ranked:
            if not self.is_blunder(self.board, move):
                return move
        
        return ranked[0][0]
    
=======
        # Tactical Filtering: Check top moves for obvious blunders
        # We check top 5 moves to find a safe one
        for move, count in ranked[:5]:
            if not self.is_blunder_tactical(self.board, move):
                return move
                
        # If all top moves are 'blunders', we might be in a bad spot anyway
        # Pick the most visited move
        return ranked[0][0]

>>>>>>> 0548ecf (First successful game played against the bot)
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
<<<<<<< HEAD
                
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
        
=======
            elif cmd == "isready":
                if self.evaluator is None: self.load_engine()
                self.send("readyok")
            elif cmd == "ucinewgame":
                self.board = chess.Board()
            elif cmd == "position":
                self.parse_position(tokens[1:])
            elif cmd == "go":
                # 1. Opening book check
                book_m = self.get_book_move()
                if book_m:
                    self.send(f"bestmove {book_m.uci()}")
                    continue
                
                # 2. Time management and search
                move = self.parse_go_and_search(tokens[1:])
                if move:
                    self.send(f"bestmove {move.uci()}")
                else:
                    legal = list(self.board.legal_moves)
                    self.send(f"bestmove {legal[0].uci() if legal else '0000'}")
            elif cmd == "quit":
                break
            elif cmd == "stop":
                pass

    def get_book_move(self):
        """Try to get a move from the polyglot book."""
        if self.book:
            try:
                with chess.polyglot.open_reader(self.book) as reader:
                    entry = reader.get(self.board)
                    if entry: return entry.move
            except: pass
        return None

    def parse_position(self, tokens):
        """Parse position command."""
        idx = 0
>>>>>>> 0548ecf (First successful game played against the bot)
        if tokens[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
<<<<<<< HEAD
            fen_parts = []
            idx += 1
=======
            idx += 1
            fen_parts = []
>>>>>>> 0548ecf (First successful game played against the bot)
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            self.board = chess.Board(" ".join(fen_parts))
        
<<<<<<< HEAD
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
=======
        if idx < len(tokens) and tokens[idx] == "moves":
            for m_uci in tokens[idx+1:]:
                self.board.push_uci(m_uci)

    def parse_go_and_search(self, tokens):
        """Calculate time limit and search."""
        wtime = 30000; btime = 30000
        movetime = None
        
        i = 0
        while i < len(tokens):
            if tokens[i] == "wtime":
                wtime = int(tokens[i+1])
                i += 2
            elif tokens[i] == "btime":
                btime = int(tokens[i+1])
                i += 2
            elif tokens[i] == "movetime":
                movetime = int(tokens[i+1])
                i += 2
            else:
                i += 1
        
        if movetime:
            # Huge buffer for safety
            limit_s = (movetime / 1000) * 0.7 
        else:
            my_time = wtime if self.board.turn == chess.WHITE else btime
            # Take 1/30th of remaining time
            limit_s = (my_time / 30000)
            
        # Buffer for tactical check and thinking overhead
        limit_s = max(0.2, limit_s - 0.3)
        # Never think for more than 15s to avoid Lichess disconnection
        limit_s = min(15.0, limit_s)
        
        return self.find_best_move(limit_s)

if __name__ == "__main__":
    FluxFishUCI().uci_loop()
>>>>>>> 0548ecf (First successful game played against the bot)
