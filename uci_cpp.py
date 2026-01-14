#!/usr/bin/env python3
"""
FluxFish UCI Interface with C++ Backend
Uses the compiled C++ engine for fast search and evaluation.
"""

import sys
import os
import time
import chess
import chess.polyglot
import subprocess
import json
from pathlib import Path

# Engine info
ENGINE_NAME = "FluxFish"
ENGINE_AUTHOR = "FluxFish Team"
MODEL_PATH = "fluxfish.bin"
BOOK_PATH = "komodo.bin"
CPP_ENGINE_PATH = "cpp/fluxfish_cpp"

# Piece values for tactical filtering
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

class FluxFishUCI:
    def __init__(self):
        self.board = chess.Board()
        self.debug = False
        self.book = None
        
    def load_engine(self):
        """Check if C++ engine is available."""
        if not os.path.exists(CPP_ENGINE_PATH):
            self.send(f"info string Error: C++ engine not found at {CPP_ENGINE_PATH}")
            return False
            
        if not os.path.exists(MODEL_PATH):
            self.send(f"info string Error: NNUE model not found at {MODEL_PATH}")
            return False
            
        self.send(f"info string Loaded C++ engine: {CPP_ENGINE_PATH}")
        if os.path.exists(BOOK_PATH):
            self.book = BOOK_PATH
            self.send(f"info string Loaded opening book: {BOOK_PATH}")
        return True
    
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
                best_recap = mat_after
                for recap in board.legal_moves:
                    if board.is_capture(recap):
                        board.push(recap)
                        best_recap = max(best_recap, self.get_material_balance(board, us))
                        board.pop()
                
                # If we lose more than 1.5 pawns worth of material
                if (mat_before - best_recap) > 150:
                    tactical_error = True
                    board.pop()
                    break
                    
                board.pop()
        
        board.pop()
        return tactical_error

    def call_cpp_engine(self, fen, time_limit_ms):
        """Call the C++ engine to get a move."""
        try:
            # Create a temporary FEN file for the C++ engine
            temp_fen = f"/tmp/fluxfish_pos_{int(time.time()*1000)}.fen"
            with open(temp_fen, 'w') as f:
                f.write(fen)
            
            # Call C++ engine with reduced iterations for faster response
            iterations = 1000 if time_limit_ms < 2000 else 3000
            
            cmd = [CPP_ENGINE_PATH, temp_fen, str(iterations)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Clean up temp file
            try:
                os.remove(temp_fen)
            except:
                pass
            
            if result.returncode == 0:
                # Parse the output to get the move
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "Best move found:" in line:
                        move_str = line.split(":")[1].strip()
                        return chess.Move.from_uci(move_str)
            
            return None
            
        except subprocess.TimeoutExpired:
            self.send("info string C++ engine timeout")
            return None
        except Exception as e:
            self.send(f"info string C++ engine error: {e}")
            return None

    def find_best_move(self, time_limit_s=None):
        """Find the best move using C++ engine and tactical filtering."""
        # Convert time limit to milliseconds
        time_limit_ms = int(time_limit_s * 1000) if time_limit_s else 2000
        
        # Get move from C++ engine
        best_move = self.call_cpp_engine(self.board.fen(), time_limit_ms)
        
        if best_move is None:
            # Fallback to first legal move if C++ engine fails
            legal_moves = list(self.board.legal_moves)
            return legal_moves[0] if legal_moves else None
        
        # Apply tactical filtering
        if not self.is_blunder_tactical(self.board, best_move):
            return best_move
        
        # If the best move is a blunder, try the next few legal moves
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves[:5]:
            if move != best_move and not self.is_blunder_tactical(self.board, move):
                self.send(f"info string Avoided blunder: {best_move.uci()}, playing: {move.uci()}")
                return move
        
        # If all moves seem like blunders, stick with the engine's choice
        self.send(f"info string All moves tactical, playing engine choice: {best_move.uci()}")
        return best_move

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
                if not self.load_engine():
                    self.send("readyok")
                    continue
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
        if tokens[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
            idx += 1
            fen_parts = []
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            self.board = chess.Board(" ".join(fen_parts))
        
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
