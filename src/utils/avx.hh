#pragma once
#include <immintrin.h>
#include "containers.hh"

//Use when size%32 != 0 for the best preformance
void quantize8_8_inplace(float* v, int16_t* q, size_t size);

int32_t dotproduct_q8_8(int16_t* w_q8_8, int16_t* x_q8_8, size_t size);

float dotproduct_fp(float* w_fp, float* x_fp, size_t size);

void update_sgd_inplace(int16_t y_hat, float y, float* w_fp, float* x_fp, size_t size, float lr);

void update_adamW_inplace(int16_t y_hat, float y, float* w_fp, float* x_fp, size_t size, AdamWParams& parmas);