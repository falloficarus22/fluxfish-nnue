import chess
import chess.engine
import chess.polyglot
import math
import numpy as np
import asyncio
import os

# --- Constants & Paths ---
STOCKFISH_PATH = "stockfish" 
FLUXFISH_PATH = "/root/fluxfish-nnue/fluxfish.sh"
BOOK_PATH = "/root/fluxfish-nnue/komodo.bin"
NUM_GAMES = 5
MOVE_TIME = 0.5  # Time per move for the players
EVAL_TIME = 0.5  # Time for Stockfish evaluation (ground truth)
WORK_DIR = "/root/fluxfish-nnue"

# --- Accuracy Formulas ---
def cp_to_win_percent(cp):
    """Win% = 50 + 50 * ( (2 / (1 + exp(-0.00368208 * cp))) - 1 )"""
    cp = max(-1000, min(1000, cp))
    return 50 + 50 * ((2 / (1 + math.exp(-0.00368208 * cp))) - 1)

def calculate_move_accuracy(win_before, win_after):
    """Accuracy% = 103.1668 * exp(-0.04354 * (win_before - win_after)) - 3.1669"""
    win_diff = max(0, win_before - win_after)
    acc = 103.1668 * math.exp(-0.04354 * win_diff) - 3.1669
    return max(0, min(100, acc))

async def evaluate_position(engine, board, time_limit):
    info = await engine.analyse(board, chess.engine.Limit(time=time_limit))
    score = info["score"].relative
    if score.is_mate():
        return 10000 if score.mate() > 0 else -10000
    return score.score()

def is_in_book(board):
    """Check if the current board position exists in the polyglot book."""
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            return reader.get(board) is not None
    except:
        return False

async def play_and_analyze(game_num):
    print(f"\n--- Starting Game {game_num + 1} ---")
    board = chess.Board()
    
    # Engines
    if game_num % 2 == 0:
        white_cmd, black_cmd = ["bash", FLUXFISH_PATH], STOCKFISH_PATH
        engine_is_white = True
    else:
        white_cmd, black_cmd = STOCKFISH_PATH, ["bash", FLUXFISH_PATH]
        engine_is_white = False

    transport_w, white_engine = await chess.engine.popen_uci(white_cmd)
    transport_b, black_engine = await chess.engine.popen_uci(black_cmd)
    transport_sv, eval_engine = await chess.engine.popen_uci(STOCKFISH_PATH)

    book_accuracies = []
    engine_accuracies = []
    out_of_book = False

    # Initial evaluation
    cp_current = await evaluate_position(eval_engine, board, EVAL_TIME)
    win_current = cp_to_win_percent(cp_current)

    try:
        while not board.is_game_over():
            win_before = win_current
            cp_before = cp_current
            
            is_engine_turn = (board.turn == chess.WHITE and engine_is_white) or \
                             (board.turn == chess.BLACK and not engine_is_white)

            # Check if engine is still in book before making move
            if is_engine_turn and not out_of_book:
                if not is_in_book(board):
                    out_of_book = True
                    print(f"info string FluxFish exited opening book at move {board.fullmove_number}")

            if board.turn == chess.WHITE:
                result = await white_engine.play(board, chess.engine.Limit(time=MOVE_TIME))
            else:
                result = await black_engine.play(board, chess.engine.Limit(time=MOVE_TIME))
            
            move = result.move
            board.push(move)

            cp_next = await evaluate_position(eval_engine, board, EVAL_TIME)
            cp_after = -cp_next
            win_after = cp_to_win_percent(cp_after)
            
            cp_current = cp_next
            win_current = cp_to_win_percent(cp_current)

            if is_engine_turn:
                acc = calculate_move_accuracy(win_before, win_after)
                if out_of_book:
                    engine_accuracies.append(acc)
                    tag = "(Neural Net)"
                else:
                    book_accuracies.append(acc)
                    tag = "(Book)"
                
                print(f"Move {board.fullmove_number}: FluxFish played {move.uci()} {tag} - Accuracy: {acc:.2f}%")
        
        avg_book = np.mean(book_accuracies) if book_accuracies else 0
        avg_engine = np.mean(engine_accuracies) if engine_accuracies else 0
        
        print(f"Game Finished. Result: {board.result()}")
        print(f"Book Accuracy: {avg_book:.2f}% ({len(book_accuracies)} moves)")
        print(f"Neural Net Accuracy: {avg_engine:.2f}% ({len(engine_accuracies)} moves)")
        
        return avg_book, avg_engine

    finally:
        await white_engine.quit()
        await black_engine.quit()
        await eval_engine.quit()

async def main():
    all_book = []
    all_engine = []
    
    for i in range(NUM_GAMES):
        book_acc, eng_acc = await play_and_analyze(i)
        if book_acc > 0: all_book.append(book_acc)
        if eng_acc > 0: all_engine.append(eng_acc)
        
        running_eng = np.mean(all_engine) if all_engine else 0
        print(f"Total Running NN Accuracy: {running_eng:.2f}%")
    
    print("\n" + "="*45)
    print(f"FINAL ACCURACY REPORT ({NUM_GAMES} GAMES)")
    print(f"Opening (Book) Accuracy:  {np.mean(all_book) if all_book else 0:.2f}%")
    print(f"Middle/End (NN) Accuracy: {np.mean(all_engine) if all_engine else 0:.2f}%")
    if all_engine:
        print(f"NN Std Deviation:         {np.std(all_engine):.2f}%")
    print("="*45)

if __name__ == "__main__":
    asyncio.run(main())
