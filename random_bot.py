import chess
import random
import sys

def main():
    board = chess.Board()
    
    # Simple UCI loop
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
            print("id name RandomBot")
            print("id author Baseline")
            print("uciok")
        elif cmd == "isready":
            print("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            # Very basic position parsing
            if "startpos" in line:
                board = chess.Board()
                if "moves" in line:
                    moves = line.split("moves ")[1].split()
                    for move in moves:
                        board.push_uci(move)
            elif "fen" in line:
                fen = line.split("fen ")[1].split(" moves")[0]
                board = chess.Board(fen)
                if "moves" in line:
                    moves = line.split("moves ")[1].split()
                    for move in moves:
                        board.push_uci(move)
        elif cmd == "go":
            # Just pick a random legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                # Prioritize captures if possible for a slightly better baseline
                captures = [m for m in legal_moves if board.is_capture(m)]
                move = random.choice(captures) if captures else random.choice(legal_moves)
                print(f"bestmove {move.uci()}")
            else:
                print("bestmove 0000")
        elif cmd == "quit":
            break

if __name__ == "__main__":
    main()
