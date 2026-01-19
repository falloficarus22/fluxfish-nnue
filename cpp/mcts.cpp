#include "mcts.h"
#include <cmath>
#include <algorithm>
#include <chrono>

// Helper to convert chess-library board to NNUE features
std::vector<float> MCTS::get_nnue_features(const chess::Board& board, chess::Color perspective) {
    std::vector<float> features(768, 0.0f);
    
    // Pieces 0-5 (Pawn to King)
    for (int pt_idx = 0; pt_idx < 6; ++pt_idx) {
        chess::PieceType pt = static_cast<chess::PieceType::underlying>(pt_idx);
        
        for (int col_idx = 0; col_idx < 2; ++col_idx) {
            chess::Color col = (col_idx == 0) ? chess::Color::WHITE : chess::Color::BLACK;
            chess::Bitboard bb = board.pieces(pt, col);
            
            while (bb) {
                int sq_idx = bb.pop();
                
                // Mirror square for black's perspective
                int final_sq = sq_idx;
                if (perspective == chess::Color::BLACK) {
                    final_sq = sq_idx ^ 56;
                }

                int rel_color = (col == perspective) ? 0 : 1;
                int idx = pt_idx * 128 + rel_color * 64 + final_sq;
                if (idx >= 0 && idx < 768) features[idx] = 1.0f;
            }
        }
    }
    return features;
}

#include <unordered_map>

// Simple evaluation cache
struct CacheEntry {
    float v;
    bool valid = false;
};
static std::unordered_map<uint64_t, CacheEntry> eval_cache;

void MCTS::select_expand_eval_backprop(chess::Board& board, MCTSNode* root) {
    MCTSNode* node = root;
    std::vector<MCTSNode*> path;
    path.push_back(node);

    // 1. Selection
    while (node->is_expanded && !node->children.empty()) {
        float best_score = -1e9f;
        MCTSNode* best_child = nullptr;

        float cpuct = 1.4f; // Slightly more focused search
        float sqrt_n = std::sqrt((float)node->n + 1e-6f);

        for (auto& child_ptr : node->children) {
            MCTSNode* child = child_ptr.get();
            
            // UCB1 formula
            float q = -child->value(); // Value from our perspective
            float u = cpuct * child->p * sqrt_n / (1.0f + child->n);
            float score = q + u;

            if (score > best_score) {
                best_score = score;
                best_child = child;
            }
        }

        if (!best_child) break;
        node = best_child;
        board.makeMove(node->move);
        path.push_back(node);
    }

    // 2. Expansion & Evaluation
    float v = 0.0f;
    auto [reason, result] = board.isGameOver();
    
    if (reason == chess::GameResultReason::NONE) {
        if (!node->is_expanded) {
            // Check cache first
            uint64_t hash = board.hash();
            if (eval_cache.count(hash)) {
                v = eval_cache[hash].v;
            } else {
                // Evaluate with NNUE
                auto w_feat = get_nnue_features(board, chess::Color::WHITE);
                auto b_feat = get_nnue_features(board, chess::Color::BLACK);
                v = nnue.evaluate(w_feat, b_feat, board.sideToMove() == chess::Color::WHITE);
                
                // Store in cache
                eval_cache[hash] = {v, true};
                if (eval_cache.size() > 100000) eval_cache.clear(); // Simple LRU-ish
            }

            // Expand
            chess::Movelist moves;
            chess::movegen::legalmoves(moves, board);
            
            if (moves.empty()) {
                v = 0.0f; // Stalemate
            } else {
                for (const auto& m : moves) {
                    float p = 1.0f / moves.size();
                    node->children.push_back(std::make_unique<MCTSNode>(m, node, p));
                }
            }
            node->is_expanded = true;
        } else {
            v = node->value(); 
        }
    } else {
        // Game Over Terminal Evaluation
        if (result == chess::GameResult::LOSE) {
            v = -1.0f; // Loss is definitively -1.0
        } else if (result == chess::GameResult::WIN) {
            v = 1.0f;
        } else {
            // Draw Aversion / Contempt
            // Penalize draws slightly to prefer playing for a win
            v = -0.05f; 
        }
    }

    // 3. Backpropagation
    // Flipping signs as we go up
    for (int i = path.size() - 1; i >= 0; --i) {
        path[i]->n++;
        path[i]->w += v;
        v = -v;
    }
}

chess::Move MCTS::search(chess::Board& board, int iterations, float time_limit_s) {
    // Clear cache for new search or keep it? Keeping it is better for iterations.
    // eval_cache.clear(); 

    MCTSNode root;
    
    // Initial expansion
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return chess::Move();

    for (const auto& m : moves) {
        root.children.push_back(std::make_unique<MCTSNode>(m, &root, 1.0f / moves.size()));
    }
    root.is_expanded = true;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        // More frequent time checks if iterations are high
        if ((i & 127) == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> elapsed = now - start_time;
            if (elapsed.count() > time_limit_s) break;
        }

        chess::Board sim_board = board;
        select_expand_eval_backprop(sim_board, &root);
    }

    // Logging bestmove info
    int max_n = -1;
    chess::Move best_move;
    for (auto& child : root.children) {
        if (child->n > max_n) {
            max_n = child->n;
            best_move = child->move;
        }
    }

    // Log search info for debugging
    if (root.n > 0) {
        std::cout << "info string nodes " << root.n << " score cp " << (int)(-root.children[0]->value() * 400) << std::endl;
    }

    return best_move;
}
