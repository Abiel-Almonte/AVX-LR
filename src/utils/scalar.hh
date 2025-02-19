//Functions with scalar implementations
#pragma once
#include <cstddef>

float dotproduct_scalar(float* w, float* x, size_t size);

float sigmoid_scalar(float x);

void update_sgd_inplace_scalar(float y_hat, float y, float* w, float* x, size_t size, float lr);