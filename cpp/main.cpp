#include <iostream>
#include <vector>
#include <fstream>
#include "chess.hpp"
#include "nnue_inference.h"
#include "mcts.h"

int main(int argc, char* argv[]) {
    std::string fen = chess::constants::STARTPOS;
    int iterations = 3000;
    
    // Parse command line arguments
    if (argc >= 2) {
        // Read FEN from file
        std::ifstream fen_file(argv[1]);
        if (fen_file.is_open()) {
            std::getline(fen_file, fen);
            fen_file.close();
        }
    }
    
    if (argc >= 3) {
        iterations = std::stoi(argv[2]);
    }
    
    // Quiet mode for UCI - only output essential info
    bool quiet_mode = (argc >= 2);
    
    if (!quiet_mode) {
        std::cout << "FluxFish C++ Engine Initializing..." << std::endl;
    }
    
    NNUE nnue;
    if (nnue.load_weights("../fluxfish.bin")) {
        if (!quiet_mode) {
            std::cout << "Successfully loaded NNUE weights." << std::endl;
        }
    } else {
        std::cerr << "Failed to load NNUE weights!" << std::endl;
        return 1;
    }

    // Initialize board with given position
    chess::Board board;
    board.setFen(fen);
    
    MCTS mcts(nnue);
    
    if (!quiet_mode) {
        std::cout << "Starting MCTS search (" << iterations << " iterations)..." << std::endl;
    }
    
    chess::Move best = mcts.search(board, iterations);
    
    std::string move_str = chess::uci::moveToUci(best);
    std::cout << "Best move found: " << move_str << std::endl;

    return 0;
}
