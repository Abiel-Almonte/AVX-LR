#pragma once

#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>

constexpr float MAXQ= 127.9940f;
constexpr float MINQ= -128.0f;
constexpr float SCALE_FACTOR= 256.0f;
constexpr float ROUND_FACTOR= 0.5f;

extern const __m256 MM256_MAXQ;
extern const __m256 MM256_MINQ;
extern const __m256 MM256_SCALE;
extern const __m256 MM256_ROUND;
extern const __m128 MM128_SCALE;
extern const __m128 MM128_ROUND;

static inline __m256 clamp(__m256 n, __m256 min, __m256 max){
    return _mm256_max_ps(min, _mm256_min_ps(max, n));
}

static inline float clamp(float n, float min, float max){
    return std::fmaxf(min, std::fminf(max, n));
}

static inline int16_t clamp(int16_t n, int16_t min, int16_t max){
    return  std::max(min, std::min(max, n));
}

static inline int16_t avx_float_to_q8_8(float n) {
    __m128 n_ps   = _mm_set_ss(n);
    __m128 result = _mm_fmadd_ss(n_ps, MM128_SCALE, MM128_ROUND);
    return static_cast<int16_t>(_mm_cvtss_si32(result));
}

constexpr inline int16_t float_to_q8_8(float n){
    return static_cast<int16_t>(n* SCALE_FACTOR + ROUND_FACTOR);
}

constexpr inline float q8_8_to_float(int16_t n){
    return static_cast<float>(n)/ SCALE_FACTOR; 
}

constexpr inline float sigmoid_fp(float x){
    return 1.0f/ (1.0f + std::exp(-x));
}

constexpr inline int16_t sigmoid_fp_to_q8_8(float x){
    return float_to_q8_8(sigmoid_fp(x));
}

constexpr inline int16_t sigmoid_q8_8(int16_t x){
    return sigmoid_fp_to_q8_8(q8_8_to_float(x));
}

constexpr std::array<int16_t, 2048> precompute_sigmoid() {
    alignas(64) std::array<int16_t, 2048> sigmoidTable{};

    sigmoidTable[0]= 0; 
    sigmoidTable[2047]= 256;
    
    for(int i= 1; i < 2047; i++){
        sigmoidTable[i]=  sigmoid_q8_8(i- 1024);
    }

    return sigmoidTable;
}
//Must use C++20
constexpr std::array<int16_t, 2048> 
SIGMOID_TABLE = precompute_sigmoid(); 

inline int16_t sigmoidApprox_fp_to_q8_8(float n){
    int16_t n_q8_8= avx_float_to_q8_8(n);
    int16_t i= clamp(n_q8_8, -1024, 1023) + 1024;
    return SIGMOID_TABLE[i];
}

inline int16_t sigmoidApprox_q8_8(int16_t n){
    int16_t i= clamp(n, -1024, 1023) + 1024;
    return SIGMOID_TABLE[i];
}

inline int16_t sigmoidApprox_q16_16_to_q8_8(int32_t n){
    int16_t n_q8_8= static_cast<int16_t>(n >> 8);
    int16_t i= clamp(n_q8_8, -1024, 1023) + 1024;
    return SIGMOID_TABLE[i];
}