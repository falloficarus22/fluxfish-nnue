#ifndef MCTS_H
#define MCTS_H

#include "chess.hpp"
#include "nnue_inference.h"
#include <vector>
#include <memory>

struct MCTSNode {
    chess::Move move;
    MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;
    
    int n = 0;
    float w = 0.0f;
    float p = 0.0f; // Prior
    bool is_expanded = false;

    // Accumulators for incremental NNUE
    Accumulator white_acc, black_acc;
    bool acc_initialized = false;

    MCTSNode(chess::Move m = chess::Move(), MCTSNode* p_parent = nullptr, float prior = 0.0f)
        : move(m), parent(p_parent), p(prior) {}
    
    float value() const { return (n == 0) ? 0.0f : w / n; }
};

class Tablebase; // Forward declare

class MCTS {
public:
    MCTS(NNUE& nnue_ref, Tablebase* tb = nullptr) : nnue(nnue_ref), tablebase(tb) {}
    chess::Move search(chess::Board& board, int iterations, float time_limit_s = -1.0f);

private:
    NNUE& nnue;
    Tablebase* tablebase;
    void select_expand_eval_backprop(chess::Board& board, MCTSNode* root);
    float get_move_priority(const chess::Board& board, chess::Move move);
    void update_node_accumulators(const chess::Board& board, chess::Move move, const Accumulator& old_w, const Accumulator& old_b, Accumulator& new_w, Accumulator& new_b);
};

#endif
