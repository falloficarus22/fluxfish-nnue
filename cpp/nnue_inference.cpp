#include "nnue_inference.h"
#include <fstream>
#include <iostream>
#include <algorithm>

NNUE::NNUE() {}

bool NNUE::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // Load binary weights exported from Python
    f.read((char*)ft_weight, sizeof(ft_weight));
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

// AVX2 optimized feature transformer forward pass
// This is the most computationally expensive part (256 * 768 multiplications)
void NNUE::ft_forward(const std::vector<float>& features, float* output) {
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        __m256 sum = _mm256_setzero_ps();
        const float* weights = ft_weight[i];
        
        // Process in chunks of 8 using AVX2
        for (int j = 0; j < NUM_FEATURES; j += 8) {
            __m256 w = _mm256_loadu_ps(&weights[j]);
            __m256 f = _mm256_loadu_ps(&features[j]);
            sum = _mm256_fmadd_ps(w, f, sum); // AVX2 FMA
        }

        // Horizontal sum of the 8 floats in __m256
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        float total_sum = 0;
        for (int k = 0; k < 8; ++k) total_sum += temp[k];
        
        output[i] = total_sum + ft_bias[i];
    }
    clipped_relu(output, HIDDEN_SIZE);
}

float NNUE::evaluate(const std::vector<float>& white_features, const std::vector<float>& black_features, bool stm) {
    float w_acc[HIDDEN_SIZE];
    float b_acc[HIDDEN_SIZE];

    ft_forward(white_features, w_acc);
    ft_forward(black_features, b_acc);

    // Concatenate according to side to move
    float input_l1[HIDDEN_SIZE * 2];
    if (stm) {
        std::copy(w_acc, w_acc + HIDDEN_SIZE, input_l1);
        std::copy(b_acc, b_acc + HIDDEN_SIZE, input_l1 + HIDDEN_SIZE);
    } else {
        std::copy(b_acc, b_acc + HIDDEN_SIZE, input_l1);
        std::copy(w_acc, w_acc + HIDDEN_SIZE, input_l1 + HIDDEN_SIZE);
    }

    // L1: Linear -> ClippedReLU
    float x1[L1_SIZE];
    for (int i = 0; i < L1_SIZE; ++i) {
        float sum = l1_bias[i];
        for (int j = 0; j < HIDDEN_SIZE * 2; ++j) {
            sum += l1_weight[i][j] * input_l1[j];
        }
        x1[i] = sum;
    }
    clipped_relu(x1, L1_SIZE);

    // L2: Linear -> ClippedReLU
    float x2[L2_SIZE];
    for (int i = 0; i < L2_SIZE; ++i) {
        float sum = l2_bias[i];
        for (int j = 0; j < L1_SIZE; ++j) {
            sum += l2_weight[i][j] * x1[j];
        }
        x2[i] = sum;
    }
    clipped_relu(x2, L2_SIZE);

    // L3: Linear -> Tanh
    float final_val = l3_bias[0];
    for (int j = 0; j < L2_SIZE; ++j) {
        final_val += l3_weight[0][j] * x2[j];
    }

    return std::tanh(final_val);
}
