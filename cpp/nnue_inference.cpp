#include "nnue_inference.h"
#include <fstream>
#include <iostream>
#include <algorithm>

NNUE::NNUE() {}

bool NNUE::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // Load binary weights exported from Python
    // Original format in file: ft_weight[HIDDEN_SIZE][NUM_FEATURES]
    float temp_ft[HIDDEN_SIZE][NUM_FEATURES];
    f.read((char*)temp_ft, sizeof(temp_ft));
    
    // Transpose to ft_weight_flat[NUM_FEATURES][HIDDEN_SIZE] for faster incremental access
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            ft_weight_flat[j * HIDDEN_SIZE + i] = temp_ft[i][j];
        }
    }

    f.read((char*)ft_bias, sizeof(ft_bias));
    f.read((char*)l1_weight, sizeof(l1_weight));
    f.read((char*)l1_bias, sizeof(l1_bias));
    f.read((char*)l2_weight, sizeof(l2_weight));
    f.read((char*)l2_bias, sizeof(l2_bias));
    f.read((char*)l3_weight, sizeof(l3_weight));
    f.read((char*)l3_bias, sizeof(l3_bias));

    return f.good();
}

void NNUE::clipped_relu(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, std::min(1.0f, data[i]));
    }
}

float NNUE::evaluate_accumulators(const Accumulator& white_acc, const Accumulator& black_acc, bool stm) {
    // 1. Activation (Clipped ReLU 0-1)
    alignas(32) float input_l1[HIDDEN_SIZE * 2];
    
    // Process white then black or vice versa
    const float* first = stm ? white_acc.vals : black_acc.vals;
    const float* second = stm ? black_acc.vals : white_acc.vals;

    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        __m256 v = _mm256_load_ps(&first[i]);
        v = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), v));
        _mm256_store_ps(&input_l1[i], v);
    }
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        __m256 v = _mm256_load_ps(&second[i]);
        v = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), v));
        _mm256_store_ps(&input_l1[HIDDEN_SIZE + i], v);
    }

    // 2. L1: Linear -> ClippedReLU
    // This is 32 x 512. We can use AVX2 to compute it.
    alignas(32) float x1[L1_SIZE];
    for (int i = 0; i < L1_SIZE; ++i) {
        __m256 sum = _mm256_setzero_ps();
        const float* weight_row = l1_weight[i];
        for (int j = 0; j < HIDDEN_SIZE * 2; j += 8) {
            __m256 w = _mm256_loadu_ps(&weight_row[j]);
            __m256 in = _mm256_load_ps(&input_l1[j]);
            sum = _mm256_fmadd_ps(w, in, sum);
        }
        // Horizontal sum
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        float row_sum = l1_bias[i];
        for (int k = 0; k < 8; ++k) row_sum += temp[k];
        x1[i] = std::max(0.0f, std::min(1.0f, row_sum));
    }

    // 3. L2: Linear -> ClippedReLU
    float x2[L2_SIZE];
    for (int i = 0; i < L2_SIZE; ++i) {
        float sum = l2_bias[i];
        for (int j = 0; j < L1_SIZE; ++j) {
            sum += l2_weight[i][j] * x1[j];
        }
        x2[i] = std::max(0.0f, std::min(1.0f, sum));
    }

    // 4. L3: Linear -> Tanh
    float final_val = l3_bias[0];
    for (int j = 0; j < L2_SIZE; ++j) {
        final_val += l3_weight[0][j] * x2[j];
    }

    // Use a faster tanh approximation if needed, but for now std::tanh is fine.
    return std::tanh(final_val);
}

float NNUE::evaluate(const std::vector<float>& white_features, const std::vector<float>& black_features, bool stm) {
    Accumulator w_acc, b_acc;
    w_acc.init(ft_bias);
    b_acc.init(ft_bias);

    for (int i = 0; i < NUM_FEATURES; ++i) {
        if (white_features[i] > 0.5f) w_acc.update(get_feature_weights(i), true);
        if (black_features[i] > 0.5f) b_acc.update(get_feature_weights(i), true);
    }

    return evaluate_accumulators(w_acc, b_acc, stm);
}
