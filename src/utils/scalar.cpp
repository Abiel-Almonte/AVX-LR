#include "scalar.hh"
#include <cmath>

float dotproduct_scalar(float* w, float* x, size_t size) {
    float prod= 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        prod+= w[i] * x[i];
    }

    return prod;
}

float sigmoid_scalar(float x){
    return 1.0f/(1.0f + std::exp(-x));
}