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

void MCTS::select_expand_eval_backprop(chess::Board& board, MCTSNode* root) {
    MCTSNode* node = root;
    std::vector<MCTSNode*> path;
    path.push_back(node);

    // 1. Selection
    while (node->is_expanded && !node->children.empty()) {
        float best_score = -1e9f;
        MCTSNode* best_child = nullptr;

        float cpuct = 1.5f;
        float sqrt_n = std::sqrt((float)node->n);

        for (auto& child_ptr : node->children) {
            MCTSNode* child = child_ptr.get();
            float u = cpuct * child->p * sqrt_n / (1.0f + child->n);
            float score = (-child->value()) + u;

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
            // Evaluate with NNUE
            auto w_feat = get_nnue_features(board, chess::Color::WHITE);
            auto b_feat = get_nnue_features(board, chess::Color::BLACK);
            v = nnue.evaluate(w_feat, b_feat, board.sideToMove() == chess::Color::WHITE);

            // Expand
            chess::Movelist moves;
            chess::movegen::legalmoves(moves, board);
            
            for (const auto& m : moves) {
                float p = 1.0f / moves.size();
                node->children.push_back(std::make_unique<MCTSNode>(m, node, p));
            }
            node->is_expanded = true;
        } else {
            v = 0.0f; 
        }
    } else {
        // Game Over Terminal Evaluation
        if (result == chess::GameResult::LOSE) {
            v = -1.0f; // Current player loses (Score relative to the player who just moved)
        } else {
            v = 0.0f; // Draw
        }
    }

    // 3. Backpropagation
    // v is the value for the player to move at 'board' (leaf)
    // We need to backpropagate it up the tree, flipping signs
    for (int i = path.size() - 1; i >= 0; --i) {
        path[i]->n++;
        path[i]->w += v;
        v = -v;
    }
}

chess::Move MCTS::search(chess::Board& board, int iterations, float time_limit_s) {
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
        if (time_limit_s > 0 && (i & 63) == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> elapsed = now - start_time;
            if (elapsed.count() > time_limit_s) break;
        }

        chess::Board sim_board = board;
        select_expand_eval_backprop(sim_board, &root);
    }

    int max_n = -1;
    chess::Move best_move;
    for (auto& child : root.children) {
        if (child->n > max_n) {
            max_n = child->n;
            best_move = child->move;
        }
    }

    return best_move;
}
