#include "logistic_regession.hh"
#include "tools.hh"
#include "avx.hh"
#include <random>

SGDLogisticRegression::SGDLogisticRegression(size_t feature_size, float learning_rate, float threshold, size_t alignment)
    :feature_size_m(feature_size),
    learning_rate_m(learning_rate),
    threshold_m(threshold),
    weights_m(alignment, feature_size),
    inputs_m(alignment, feature_size),
    weights_q8_8_m(alignment, feature_size),
    inputs_q8_8_m(alignment, feature_size){
    initWeights();
}

void SGDLogisticRegression::setThreshold(float threshold){
    threshold_m= threshold; 
}

void SGDLogisticRegression::setLearningRate(float learning_rate){
    learning_rate_m= learning_rate;
}

void SGDLogisticRegression::setInputs(float* x){
    size_t i= 0;
    for (; i + 16 <= feature_size_m; i += 16){
        _mm_prefetch(reinterpret_cast<const char*>(&x[i + 32]), _MM_HINT_T0);

        __m256 vec1= _mm256_loadu_ps(&x[i]);
        __m256 vec2= _mm256_loadu_ps(&x[i + 8]);

        _mm256_store_ps(&inputs_m[i], vec1);
        _mm256_store_ps(&inputs_m[i+ 8], vec2);
    }

    for(; i < feature_size_m; i++ ){
        inputs_m[i]= x[i];
    }
}

int16_t SGDLogisticRegression::inference_q8_8(alignedArray<float>& inputs){
    quantize8_8_inplace(inputs.data(), inputs_q8_8_m.data(), feature_size_m);
    return sigmoidApprox_q16_16_to_q8_8(dotproduct_q8_8(weights_q8_8_m.data(), inputs_q8_8_m.data(), feature_size_m));
}

float SGDLogisticRegression::inference_fp(alignedArray<float>& inputs){
    return q8_8_to_float(sigmoid_fp_to_q8_8(dotproduct_fp(weights_m.data(), inputs.data(), feature_size_m)));
}

float SGDLogisticRegression::inference_q8_8_to_fp(alignedArray<float>& inputs){
    return q8_8_to_float(inference_q8_8(inputs));
}

void SGDLogisticRegression::update_weights(float prediction, float label) {
    sgd_inplace(prediction, label, weights_m.data(), inputs_m.data(), feature_size_m, learning_rate_m);
    quantize8_8_inplace(weights_m.data(), weights_q8_8_m.data(), feature_size_m);
}

//xavier init
void SGDLogisticRegression::initWeights(){
    std::random_device rd;
    std::mt19937 gen(rd());

    float limit= sqrt(6.0f / feature_size_m);
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (size_t i = 0; i < feature_size_m; i++) {
        weights_m[i]= dist(gen);
    }

    quantize8_8_inplace(weights_m.data(), weights_q8_8_m.data(), feature_size_m);
}
