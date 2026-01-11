import chess
from evaluate import Evaluator
from mcts import MCTS

def main():
    # Path to the file
    engine_brain = Evaluator("nn-1c0000000000.nnue")
    searcher = MCTS(engine_brain.evaluate)
    board = chess.Board()

    while not board.is_game_over():
        print(board)

        if board.turn == chess.WHITE:
            move = input("Your Move (UCI): ")

            try:
                board.push_uci(move)
            except:
                print("Invalid move")
                continue

        else:
            print("Engine Thinking...")
            best_move = searcher.search(board, iterations = 300)
            print(f"Engine Played: {best_move}")
            board.push(best_move)

if __name__ == "__main__":
    main()
    