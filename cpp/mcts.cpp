#include "mcts.h"
#include "tablebase.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <unordered_map>

// Helper for NNUE feature indexing
inline int get_feature_index(int sq, int pt, int pc, int perspective) {
    int final_sq = (perspective == (int)chess::Color::BLACK) ? sq ^ 56 : sq;
    int rel_color = (pc == perspective) ? 0 : 1;
    return pt * 128 + rel_color * 64 + final_sq;
}

// Tactical heuristic values
static const int PIECE_VALS[] = { 100, 320, 330, 500, 900, 0 };

float MCTS::get_move_priority(const chess::Board& board, chess::Move move) {
    float score = 1.0f;
    
    // 1. Capture (MVV-LVA)
    chess::Piece captured = board.at<chess::Piece>(move.to());
    if (captured != chess::Piece::NONE) {
        chess::Piece attacker = board.at<chess::Piece>(move.from());
        score += 10.0f + (PIECE_VALS[(int)captured.type()] / 10.0f) - (PIECE_VALS[(int)attacker.type()] / 100.0f);
    }
    
    // 2. Promotion
    if (move.typeOf() == chess::Move::PROMOTION) {
        score += 8.0f;
    }
    
    // 3. Check
    if (board.givesCheck(move) != chess::CheckType::NO_CHECK) {
        score += 2.0f;
    }

    // 4. Center control
    int to_sq = move.to().index();
    if (to_sq == 27 || to_sq == 28 || to_sq == 35 || to_sq == 36) { // d4, e4, d5, e5
        score += 0.5f;
    }

    return score;
}

void MCTS::update_node_accumulators(const chess::Board& board, chess::Move move, const Accumulator& old_w, const Accumulator& old_b, Accumulator& new_w, Accumulator& new_b) {
    new_w = old_w;
    new_b = old_b;

    chess::Piece moving_p = board.at<chess::Piece>(move.from());
    chess::Piece captured_p = board.at<chess::Piece>(move.to());
    
    auto update = [&](int sq, chess::Piece p, bool added) {
        int pt = (int)p.type();
        int pc = (int)p.color();
        new_w.update(nnue.get_feature_weights(get_feature_index(sq, pt, pc, (int)chess::Color::WHITE)), added);
        new_b.update(nnue.get_feature_weights(get_feature_index(sq, pt, pc, (int)chess::Color::BLACK)), added);
    };

    update(move.from().index(), moving_p, false);
    
    if (move.typeOf() == chess::Move::ENPASSANT) {
        int cap_sq = move.to().index() ^ 8;
        update(cap_sq, board.at<chess::Piece>(cap_sq), false);
    } else if (captured_p != chess::Piece::NONE) {
        update(move.to().index(), captured_p, false);
    }

    if (move.typeOf() == chess::Move::PROMOTION) {
        chess::Piece promo_p = chess::Piece(move.promotionType(), moving_p.color());
        update(move.to().index(), promo_p, true);
    } else {
        update(move.to().index(), moving_p, true);
    }

    if (move.typeOf() == chess::Move::CASTLING) {
        int r_from, r_to;
        if (move.to() == chess::Square::SQ_G1) { r_from = 7; r_to = 5; }
        else if (move.to() == chess::Square::SQ_C1) { r_from = 0; r_to = 3; }
        else if (move.to() == chess::Square::SQ_G8) { r_from = 63; r_to = 61; }
        else if (move.to() == chess::Square::SQ_C8) { r_from = 56; r_to = 59; }
        else { return; }
        
        chess::Piece rook = board.at<chess::Piece>(r_from);
        update(r_from, rook, false);
        update(r_to, rook, true);
    }
}

// Simple evaluation cache
struct CacheEntry {
    float v;
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

        float cpuct = 1.6f; 
        float sqrt_n = std::sqrt((float)node->n + 1e-6f);

        for (auto& child_ptr : node->children) {
            MCTSNode* child = child_ptr.get();
            float q = -child->value(); 
            float u = cpuct * child->p * sqrt_n / (1.0f + child->n);
            float score = q + u;

            if (score > best_score) {
                best_score = score;
                best_child = child;
            }
        }

        if (!best_child) break;

        // Lazy compute accumulator for the selected child
        if (!best_child->acc_initialized) {
            update_node_accumulators(board, best_child->move, node->white_acc, node->black_acc, best_child->white_acc, best_child->black_acc);
            best_child->acc_initialized = true;
        }

        board.makeMove(best_child->move);
        node = best_child;
        path.push_back(node);
    }

    // 2. Expansion & Evaluation
    float v = 0.0f;
    bool tb_hit = false;
    
    if (!node->is_expanded) {
        // Generate moves once
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);
        
        if (moves.empty()) {
            // Game Over
            if (board.inCheck()) {
                v = -1.0f; // Checkmate
            } else {
                v = -0.02f; // Stalemate / Draw contempt
            }
        } else {
            // Try tablebase probe first
            if (tablebase != nullptr && tablebase->can_probe(board)) {
                int tb_result = tablebase->probe_wdl(board);
                if (tb_result != TB_RESULT_FAILED) {
                    tb_hit = true;
                    // Convert TB result to evaluation
                    // TB_WIN = 2, TB_CURSED_WIN = 1, TB_DRAW = 0, TB_BLESSED_LOSS = -1, TB_LOSS = -2
                    switch (tb_result) {
                        case TB_WIN:
                        case TB_CURSED_WIN:
                            v = 0.95f; // Slight less than mate to prefer faster wins
                            break;
                        case TB_DRAW:
                            v = -0.02f; // Small contempt
                            break;
                        case TB_BLESSED_LOSS:
                        case TB_LOSS:
                            v = -0.95f; // Slight less than mate
                            break;
                        default:
                            tb_hit = false;
                    }
                    
                    if (tb_hit) {
                        // Don't expand tablebase positions - they're terminal
                        node->is_expanded = true;
                    }
                }
            }
            
            if (!tb_hit) {
                // Check cache
                uint64_t hash = board.hash();
                if (eval_cache.count(hash)) {
                    v = eval_cache[hash].v;
                } else {
                    // Root is always initialized, others initialized during selection
                    v = nnue.evaluate_accumulators(node->white_acc, node->black_acc, board.sideToMove() == chess::Color::WHITE);
                    eval_cache[hash] = {v};
                    if (eval_cache.size() > 200000) eval_cache.clear();
                }

                // Expand - Minimal work here
                float total_priority = 0.0f;
                std::vector<float> priorities;
                for (const auto& m : moves) {
                    float prio = get_move_priority(board, m);
                    priorities.push_back(prio);
                    total_priority += prio;
                }

                for (size_t j = 0; j < (size_t)moves.size(); ++j) {
                    float p = priorities[j] / total_priority;
                    // No accumulator update here! (Lazy Expansion)
                    node->children.push_back(std::make_unique<MCTSNode>(moves[j], node, p));
                }
                node->is_expanded = true;
            }
        }
    } else {
        v = node->value();
    }

    // 3. Backpropagation & Unmake
    for (int i = (int)path.size() - 1; i >= 0; --i) {
        path[i]->n++;
        path[i]->w += v;
        v = -v;
        if (i > 0) board.unmakeMove(path[i]->move);
    }
}

chess::Move MCTS::search(chess::Board& board, int iterations, float time_limit_s) {
    MCTSNode root;
    
    // Initialize root accumulators
    root.white_acc.init(nnue.get_ft_bias());
    root.black_acc.init(nnue.get_ft_bias());
    for (int sq = 0; sq < 64; ++sq) {
        chess::Square square = chess::Square(sq);
        chess::Piece p = board.at<chess::Piece>(square);
        if (p != chess::Piece::NONE) {
            int pt = (int)p.type();
            int pc = (int)p.color();
            root.white_acc.update(nnue.get_feature_weights(get_feature_index(sq, pt, pc, (int)chess::Color::WHITE)), true);
            root.black_acc.update(nnue.get_feature_weights(get_feature_index(sq, pt, pc, (int)chess::Color::BLACK)), true);
        }
    }
    root.acc_initialized = true;

    // Initial expansion
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return chess::Move();

    float total_priority = 0.0f;
    std::vector<float> priorities;
    for (const auto& m : moves) {
        float prio = get_move_priority(board, m);
        priorities.push_back(prio);
        total_priority += prio;
    }

    for (size_t i = 0; i < (size_t)moves.size(); ++i) {
        float p = priorities[i] / total_priority;
        root.children.push_back(std::make_unique<MCTSNode>(moves[i], &root, p));
    }
    root.is_expanded = true;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if ((i & 255) == 0 && time_limit_s > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> elapsed = now - start_time;
            if (elapsed.count() > time_limit_s) break;
        }
        select_expand_eval_backprop(board, &root);
    }

    int max_n = -1;
    chess::Move best_move;
    for (auto& child : root.children) {
        if (child->n > max_n) {
            max_n = child->n;
            best_move = child->move;
        }
    }

    if (root.n > 0) {
        std::cout << "info string nodes " << root.n << " nps " << (int)(root.n / ((std::chrono::duration<float>)(std::chrono::high_resolution_clock::now() - start_time)).count()) << std::endl;
    }

    return best_move;
}
