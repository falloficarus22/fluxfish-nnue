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
        self.cpp_process = None
        
    def load_engine(self):
        """Start the persistent C++ engine process."""
        if not os.path.exists(CPP_ENGINE_PATH):
            self.send(f"info string Error: C++ engine not found at {CPP_ENGINE_PATH}")
            return False
            
        try:
            # Start C++ engine in UCI mode
            self.cpp_process = subprocess.Popen(
                [CPP_ENGINE_PATH, "uci"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1 # Line buffered
            )
            
            # Send initial UCI commands
            self.send_to_cpp("uci")
            
            # Wait for uciok
            while True:
                line = self.cpp_process.stdout.readline().strip()
                if line == "uciok":
                    break
            
            if os.path.exists(BOOK_PATH):
                self.book = BOOK_PATH
                self.send(f"info string Loaded opening book: {BOOK_PATH}")
                
            return True
        except Exception as e:
            self.send(f"info string Error starting C++ engine: {e}")
            return False
    
    def send(self, msg):
        """Send a message to the UCI GUI."""
        print(msg, flush=True)

    def send_to_cpp(self, msg):
        """Send a message to the C++ engine's stdin."""
        if self.cpp_process and self.cpp_process.poll() is None:
            self.cpp_process.stdin.write(msg + "\n")
            self.cpp_process.stdin.flush()

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
                if not self.cpp_process:
                    if not self.load_engine():
                        self.send("readyok")
                        continue
                self.send_to_cpp("isready")
                # Wait for readyok from CPP
                while True:
                    cpp_line = self.cpp_process.stdout.readline().strip()
                    if cpp_line == "readyok":
                        self.send("readyok")
                        break
                    elif "info" in cpp_line:
                        self.send(cpp_line)
            elif cmd == "ucinewgame":
                self.board = chess.Board()
                self.send_to_cpp("ucinewgame")
            elif cmd == "position":
                self.parse_position(tokens[1:])
                # Forward the exact position command to C++
                self.send_to_cpp(line)
            elif cmd == "go":
                # 1. Opening book check
                book_m = self.get_book_move()
                if book_m:
                    self.send(f"bestmove {book_m.uci()}")
                    continue
                
                # 2. Forward go command to C++
                self.send_to_cpp(line)
                
                # 3. Read info and bestmove from C++
                while True:
                    cpp_line = self.cpp_process.stdout.readline().strip()
                    if not cpp_line:
                        break
                    if cpp_line.startswith("bestmove"):
                        self.send(cpp_line)
                        break
                    else:
                        # Forward info strings to GUI
                        self.send(cpp_line)
                        
            elif cmd == "quit":
                self.send_to_cpp("quit")
                if self.cpp_process:
                    self.cpp_process.terminate()
                break
            elif cmd == "stop":
                self.send_to_cpp("stop")

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
        """Sync our local board state (mostly for book usage)."""
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

if __name__ == "__main__":
    FluxFishUCI().uci_loop()
