#ifndef NNUE_INFERENCE_H
#define NNUE_INFERENCE_H

#include <vector>
#include <string>
#include <cmath>
#include <immintrin.h> // For AVX2
#include <algorithm>

const int NUM_FEATURES = 768; // 6 * 2 * 64
const int HIDDEN_SIZE = 256;
const int L1_SIZE = 32;
const int L2_SIZE = 32;

struct Accumulator {
    alignas(32) float vals[HIDDEN_SIZE];

    void init(const float* bias) {
        std::copy(bias, bias + HIDDEN_SIZE, vals);
    }

    void update(const float* weights, bool added) {
        if (added) {
            for (int i = 0; i < HIDDEN_SIZE; i += 8) {
                __m256 v = _mm256_load_ps(&vals[i]);
                __m256 w = _mm256_loadu_ps(&weights[i]);
                _mm256_store_ps(&vals[i], _mm256_add_ps(v, w));
            }
        } else {
            for (int i = 0; i < HIDDEN_SIZE; i += 8) {
                __m256 v = _mm256_load_ps(&vals[i]);
                __m256 w = _mm256_loadu_ps(&weights[i]);
                _mm256_store_ps(&vals[i], _mm256_sub_ps(v, w));
            }
        }
    }
};

class NNUE {
public:
    NNUE();
    bool load_weights(const std::string& path);
    
    // Legacy evaluation
    float evaluate(const std::vector<float>& white_features, const std::vector<float>& black_features, bool stm);

    // Incremental evaluation
    float evaluate_accumulators(const Accumulator& white_acc, const Accumulator& black_acc, bool stm);
    
    const float* get_feature_weights(int feature_idx) const {
        return ft_weight_flat + (feature_idx * HIDDEN_SIZE);
    }
    
    const float* get_ft_bias() const { return ft_bias; }

private:
    // Feature Transformer weights - flattened for better access
    // We want weights[feature][hidden] for incremental updates
    alignas(32) float ft_weight_flat[NUM_FEATURES * HIDDEN_SIZE];
    alignas(32) float ft_bias[HIDDEN_SIZE];

    // Layer 1
    float l1_weight[L1_SIZE][HIDDEN_SIZE * 2];
    float l1_bias[L1_SIZE];

    // Layer 2
    float l2_weight[L2_SIZE][L1_SIZE];
    float l2_bias[L2_SIZE];

    // Layer 3
    float l3_weight[1][L2_SIZE];
    float l3_bias[1];

    // Helper for Clipped ReLU
    void clipped_relu(float* data, int size);
};

#endif
