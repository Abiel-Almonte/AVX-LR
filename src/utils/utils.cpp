#include "containers.hh"
#include "tools.hh"
#include <immintrin.h>

alignas(32) const __m256 MM256_MAXQ= _mm256_broadcast_ss(&MAXQ);
alignas(32) const __m256 MM256_MINQ= _mm256_broadcast_ss(&MINQ);
alignas(32) const __m256 MM256_SCALE= _mm256_broadcast_ss(&SCALE_FACTOR);
alignas(32) const __m256 MM256_ROUND= _mm256_broadcast_ss(&ROUND_FACTOR);
alignas(32) const __m128 MM128_SCALE= _mm_broadcast_ss(&SCALE_FACTOR);
alignas(32) const __m128 MM128_ROUND= _mm_broadcast_ss(&ROUND_FACTOR);

AdamWParams::AdamWParams(size_t size, float lr):
        learning_rate(lr), 
        beta1(0.9f),
        beta2(0.999f),
        beta1_complement(0.1f),
        beta2_complement(1e-3f),
        beta1i(0.9f), 
        beta2i(0.999f),
        decay(0.99999f), 
        eps(1e-8f),
        moment1(size), 
        moment2(size)
    {   
        __m256 zeros= _mm256_setzero_ps();

        size_t i= 0;
        for(; i <= size; i+= 8){
            _mm256_store_ps(moment1.data() + i, zeros);
            _mm256_store_ps(moment2.data() + i, zeros);
        }

        for(; i < size; i++){
            moment1[i]= 0.0f;   
            moment2[i]= 0.0f;
        }
    } 

