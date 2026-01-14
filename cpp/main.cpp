#include <iostream>
#include <vector>
#include "chess.hpp"
#include "nnue_inference.h"
#include "mcts.h"

int main() {
    std::cout << "FluxFish C++ Engine Initializing..." << std::endl;
    
    NNUE nnue;
    if (nnue.load_weights("../fluxfish.bin")) {
        std::cout << "Successfully loaded NNUE weights." << std::endl;
    } else {
        std::cerr << "Failed to load NNUE weights!" << std::endl;
        return 1;
    }

    // Initialize board with starting position
    chess::Board board;
    board.setFen(chess::constants::STARTPOS);
    
    MCTS mcts(nnue);
    
    std::cout << "Starting MCTS search (10000 iterations)..." << std::endl;
    chess::Move best = mcts.search(board, 10000);
    
    std::string move_str = chess::uci::moveToUci(best);
    std::cout << "Best move found: " << move_str << std::endl;

    return 0;
}
