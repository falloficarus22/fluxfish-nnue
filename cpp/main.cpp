#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <thread>
#include <atomic>
#include <algorithm>
#include "chess.hpp"
#include "nnue_inference.h"
#include "mcts.h"

// Simple time management
struct SearchParams {
    int wtime = 0;
    int btime = 0;
    int winc = 0;
    int binc = 0;
    int movetime = 0;
    bool infinite = false;
};

void uci_loop() {
    chess::Board board;
    NNUE nnue;
    
    // Try to load weights from multiple locations
    if (nnue.load_weights("fluxfish.bin")) {
        std::cout << "info string Loaded weights from fluxfish.bin" << std::endl;
    } else if (nnue.load_weights("../fluxfish.bin")) {
        std::cout << "info string Loaded weights from ../fluxfish.bin" << std::endl;
    } else {
        std::cout << "info string Failed to load NNUE weights! Engine will plain random moves/crash." << std::endl;
    }
    
    MCTS mcts(nnue);
    
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        
        if (token == "uci") {
            std::cout << "id name FluxFish C++" << std::endl;
            std::cout << "id author FluxFish Team" << std::endl;
            std::cout << "option name Hash type spin default 16 min 1 max 1024" << std::endl;
            std::cout << "option name Threads type spin default 1 min 1 max 1" << std::endl;
            std::cout << "uciok" << std::endl;
        }
        else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        }
        else if (token == "ucinewgame") {
            board = chess::Board();
            // Clear MCTS tree/cache if implemented
        }
        else if (token == "position") {
            std::string arg;
            iss >> arg;
            if (arg == "startpos") {
                board = chess::Board();
                if (iss >> arg && arg == "moves") {
                    // consume "moves" token, proceed to loop
                }
            } else if (arg == "fen") {
                std::string fen;
                while (iss >> arg && arg != "moves") {
                    fen += arg + " ";
                }
                board.setFen(fen);
            }
            
            // Apply moves
            // If the last token was "moves" or we are subsequent
            if (arg == "moves") {
                std::string move_str;
                while (iss >> move_str) {
                    // chess::uci::parseSan is not standard, use uci::parseUci or manual
                    // The library likely supports board.makeMove(chess::uci::uciToMove(board, move_str));
                    // Let's assume uci::uciToMove exists (standard name).
                    // If not, we fix it in compilation.
                    try {
                        chess::Move move = chess::uci::uciToMove(board, move_str);
                        board.makeMove(move);
                    } catch (...) {
                       // ignore illegal
                    }
                }
            }
        }
        else if (token == "go") {
            SearchParams params;
            std::string arg;
            while (iss >> arg) {
                if (arg == "wtime") { iss >> params.wtime; }
                else if (arg == "btime") { iss >> params.btime; }
                else if (arg == "winc") { iss >> params.winc; }
                else if (arg == "binc") { iss >> params.binc; }
                else if (arg == "movetime") { iss >> params.movetime; }
                else if (arg == "infinite") { params.infinite = true; }
            }
            
            // Time Management
            float time_limit_s = 5.0f; // Default for infinite/ponder
            int iterations = 2000000; // Limit iterations if needed
            
            if (params.movetime > 0) {
                time_limit_s = (params.movetime / 1000.0f) * 0.95f; 
            } else if (params.wtime > 0) {
                // Simple 1/30th time management
                int my_time = (board.sideToMove() == chess::Color::WHITE) ? params.wtime : params.btime;
                int my_inc = (board.sideToMove() == chess::Color::WHITE) ? params.winc : params.binc;
                // Use slightly more efficient time mgmt: time / 20 + inc
                time_limit_s = (my_time / 20.0f + my_inc * 0.5f) / 1000.0f;
            }
            
            // Buffer to avoid loss on time
            time_limit_s = std::max(0.01f, time_limit_s - 0.05f); 
            
            chess::Move best = mcts.search(board, iterations, time_limit_s);
            std::cout << "bestmove " << chess::uci::moveToUci(best) << std::endl;
        }
        else if (token == "quit") {
            break;
        }
    }
}

#include <fstream>

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return str;
}

int main(int argc, char* argv[]) {
    // Check if arguments provided (CLI mode)
    if (argc >= 3) {
        std::string fen_path = argv[1];
        int iterations = std::stoi(argv[2]);
        
        std::string fen = read_file(fen_path);
        if (fen.empty()) {
            std::cerr << "Error: Could not read FEN from " << fen_path << std::endl;
            return 1;
        }

        chess::Board board;
        board.setFen(fen);
        
        NNUE nnue;
        if (!nnue.load_weights("fluxfish.bin") && !nnue.load_weights("../fluxfish.bin")) {
             // Continue anyway, but it will play poorly
        }

        MCTS mcts(nnue);
        chess::Move best = mcts.search(board, iterations, 10.0f); // 10s max safe limit for CLI
        
        std::cout << "Best move found: " << chess::uci::moveToUci(best) << std::endl;
        return 0;
    }
    
    // Check for "uci" flag for standard UCI mode
    if (argc == 2 && std::string(argv[1]) == "uci") {
        uci_loop();
        return 0;
    }
    
    // Default to UCI loop for interactive use
    uci_loop();

    return 0;
}
