import chess
import chess.engine
import math
import numpy as np
import asyncio
import os

# --- Constants & Paths ---
STOCKFISH_PATH = "stockfish" 
FLUXFISH_PATH = "/root/fluxfish-nnue/fluxfish.sh"
NUM_GAMES = 5
EVAL_DEPTH = 14  # Depth for Stockfish evaluation
WORK_DIR = "/root/fluxfish-nnue"

# --- Accuracy Formulas ---
def cp_to_win_percent(cp):
    """Win% = 50 + 50 * ( (2 / (1 + exp(-0.00368208 * cp))) - 1 )"""
    # Clamp CP to avoid overflow in exp
    cp = max(-1000, min(1000, cp))
    return 50 + 50 * ((2 / (1 + math.exp(-0.00368208 * cp))) - 1)

def calculate_move_accuracy(win_before, win_after):
    """Accuracy% = 103.1668 * exp(-0.04354 * (win_before - win_after)) - 3.1669"""
    # win_diff is always positive for a drop in accuracy (how much win% we lost)
    # If win_after > win_before (blunder by opponent or find better move than engine expected), win_diff is negative
    # usually we cap the difference at 0 to avoid > 100% accuracy
    win_diff = max(0, win_before - win_after)
    acc = 103.1668 * math.exp(-0.04354 * win_diff) - 3.1669
    return max(0, min(100, acc))

async def evaluate_position(engine, board, depth):
    info = await engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].relative
    if score.is_mate():
        return 10000 if score.mate() > 0 else -10000
    return score.score()

async def play_and_analyze(game_num):
    print(f"\n--- Starting Game {game_num + 1} ---")
    board = chess.Board()
    
    # Engines
    # FluxFish plays White in even games, Black in odd
    if game_num % 2 == 0:
        white_cmd, black_cmd = ["bash", FLUXFISH_PATH], STOCKFISH_PATH
        engine_is_white = True
    else:
        white_cmd, black_cmd = STOCKFISH_PATH, ["bash", FLUXFISH_PATH]
        engine_is_white = False

    transport_w, white_engine = await chess.engine.popen_uci(white_cmd)
    transport_b, black_engine = await chess.engine.popen_uci(black_cmd)
    
    # Analysis engine (always SF)
    transport_sv, eval_engine = await chess.engine.popen_uci(STOCKFISH_PATH)

    move_accuracies = []
    
    try:
        while not board.is_game_over():
            # Evaluation BEFORE move (relative to player whose turn it is)
            cp_before = await evaluate_position(eval_engine, board, EVAL_DEPTH)
            win_before = cp_to_win_percent(cp_before)
            
            is_engine_turn = (board.turn == chess.WHITE and engine_is_white) or \
                             (board.turn == chess.BLACK and not engine_is_white)

            if board.turn == chess.WHITE:
                result = await white_engine.play(board, chess.engine.Limit(time=0.5))
            else:
                result = await black_engine.play(board, chess.engine.Limit(time=0.5))
            
            move = result.move
            board.push(move)

            # Evaluation AFTER move (relative to the SAME player who just moved)
            # board.turn has swapped, so we need to negate the score to get it relative to the mover
            cp_after_rel = await evaluate_position(eval_engine, board, EVAL_DEPTH)
            cp_after = -cp_after_rel 
            win_after = cp_to_win_percent(cp_after)

            if is_engine_turn:
                acc = calculate_move_accuracy(win_before, win_after)
                move_accuracies.append(acc)
                print(f"Move {board.fullmove_number}: Engine played {move.uci()} - Accuracy: {acc:.2f}% (CP: {cp_before} -> {cp_after})")
        
        game_acc = np.mean(move_accuracies) if move_accuracies else 0
        print(f"Game Finished. Result: {board.result()}. Engine Avg Accuracy: {game_acc:.2f}%")
        return game_acc

    finally:
        await white_engine.quit()
        await black_engine.quit()
        await eval_engine.quit()

async def main():
    total_accuracies = []
    for i in range(NUM_GAMES):
        acc = await play_and_analyze(i)
        total_accuracies.append(acc)
        print(f"Current Running Average Accuracy: {np.mean(total_accuracies):.2f}%")
    
    print("\n" + "="*40)
    print(f"FINAL ESTIMATED POWER (5 GAMES)")
    print(f"Mean Accuracy: {np.mean(total_accuracies):.2f}%")
    print(f"Std Deviation: {np.std(total_accuracies):.2f}%")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
