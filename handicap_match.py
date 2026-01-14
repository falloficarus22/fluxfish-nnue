import chess
import chess.engine
import asyncio
import os

# --- Configuration ---
FLUXFISH_PATH = "/root/fluxfish-nnue/fluxfish.sh"
STOCKFISH_PATH = ["python3", "/root/fluxfish-nnue/random_bot.py"]
FLUX_TIME = 1.0    
SF_TIME = 0.1      
NUM_GAMES = 3
# No Elo limiting needed for RandomBot

async def play_game(game_num):
    board = chess.Board()
    
    # Alternate colors for fairness
    if game_num % 2 == 0:
        white_cmd, black_cmd = ["bash", FLUXFISH_PATH], STOCKFISH_PATH
        white_name, black_name = "FluxFish", "Stockfish"
        white_time, black_time = FLUX_TIME, SF_TIME
    else:
        white_cmd, black_cmd = STOCKFISH_PATH, ["bash", FLUXFISH_PATH]
        white_name, black_name = "Stockfish", "FluxFish"
        white_time, black_time = SF_TIME, FLUX_TIME

    print(f"\n--- Game {game_num + 1}: {white_name} (W) vs {black_name} (B) ---")
    
    # Start engines
    transport_w, white_engine = await chess.engine.popen_uci(white_cmd)
    transport_b, black_engine = await chess.engine.popen_uci(black_cmd)

    try:
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                limit = chess.engine.Limit(time=white_time)
                result = await white_engine.play(board, limit)
            else:
                limit = chess.engine.Limit(time=black_time)
                result = await black_engine.play(board, limit)
            
            board.push(result.move)
            
            # Print move and current board evaluation for context (silent)
            # print(f"{board.fullmove_number}.{ '..' if board.turn == chess.WHITE else ''} {result.move}", end=" ", flush=True)

        print(f"\nResult: {board.result()}")
        return board.result(), white_name, black_name

    finally:
        await white_engine.quit()
        await black_engine.quit()

async def main():
    flux_score = 0
    sf_score = 0
    draws = 0

    print(f"Starting Match: FluxFish vs RandomBot")
    print(f"Total Games: {NUM_GAMES}\n")

    for i in range(NUM_GAMES):
        result, white_name, black_name = await play_game(i)
        
        if result == "1-0":
            if white_name == "FluxFish": flux_score += 1
            else: sf_score += 1
        elif result == "0-1":
            if black_name == "FluxFish": flux_score += 1
            else: sf_score += 1
        else:
            flux_score += 0.5
            sf_score += 0.5
            draws += 1
        
        print(f"Match Score: FluxFish {flux_score} - Stockfish {sf_score} (Draws: {draws})")
        print("-" * 30)

    print("\n" + "="*40)
    print("FINAL HANDICAP MATCH RESULTS")
    print(f"FluxFish:  {flux_score} ({FLUX_TIME}s/move)")
    print(f"Stockfish: {sf_score} ({SF_TIME}s/move)")
    print(f"Draws:     {draws}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
