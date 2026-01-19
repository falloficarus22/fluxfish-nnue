import chess
import chess.engine
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def play_game(white_engine, black_engine, time_limit=1.0):
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            result = await white_engine.play(board, chess.engine.Limit(time=time_limit))
        else:
            result = await black_engine.play(board, chess.engine.Limit(time=time_limit))
        
        board.push(result.move)
        # print(".", end="", flush=True)
    
    # print()
    return board.result()

async def run_match(num_games=10, time_per_move=0.5):
    fluxfish_path = "/root/fluxfish-nnue/fluxfish.sh"
    stockfish_path = "stockfish"

    logger.info(f"Starting match: FluxFish vs Stockfish")
    logger.info(f"Games: {num_games}, Time per move: {time_per_move}s")

    flux_score = 0
    sf_score = 0
    draws = 0

    for i in range(num_games):
        # Alternate colors
        if i % 2 == 0:
            white_path, black_path = fluxfish_path, stockfish_path
            white_name, black_name = "FluxFish", "Stockfish"
        else:
            white_path, black_path = stockfish_path, fluxfish_path
            white_name, black_name = "Stockfish", "FluxFish"

        transport1, engine1 = await chess.engine.popen_uci(["bash", white_path] if white_path.endswith(".sh") else white_path)
        transport2, engine2 = await chess.engine.popen_uci(["bash", black_path] if black_path.endswith(".sh") else black_path)

        logger.info(f"Game {i+1}: {white_name} (W) vs {black_name} (B)...")
        
        try:
            result = await play_game(engine1, engine2, time_limit=time_per_move)
            logger.info(f"Result: {result}")

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
        except Exception as e:
            logger.error(f"Error in game {i+1}: {e}")
        finally:
            await engine1.quit()
            await engine2.quit()

        logger.info(f"Current Score: FluxFish {flux_score} - Stockfish {sf_score} (Draws: {draws})")
        logger.info("-" * 20)

    logger.info("Match Finished!")
    logger.info(f"Final Score: FluxFish {flux_score} - Stockfish {sf_score}")
    logger.info(f"Draws: {draws}")

if __name__ == "__main__":
    asyncio.run(run_match(num_games=5, time_per_move=1.0))
