#ifndef NNUE_INFERENCE_H
#define NNUE_INFERENCE_H

#include <vector>
#include <string>
#include <cmath>
#include <immintrin.h> // For AVX2

const int NUM_FEATURES = 768; // 6 * 2 * 64
const int HIDDEN_SIZE = 256;
const int L1_SIZE = 32;
const int L2_SIZE = 32;

class NNUE {
public:
    NNUE();
    bool load_weights(const std::string& path);
    float evaluate(const std::vector<float>& white_features, const std::vector<float>& black_features, bool stm);

private:
    // Feature Transformer weights
    float ft_weight[HIDDEN_SIZE][NUM_FEATURES];
    float ft_bias[HIDDEN_SIZE];

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
    
    // AVX2 optimized dot product for the large feature transformer
    void ft_forward(const std::vector<float>& features, float* output);
};

#endif
