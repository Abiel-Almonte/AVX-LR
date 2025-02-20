#include "tools.hh"
#include "avx.hh"

void quantize8_8_inplace(float* v, int16_t* q, size_t size){
    size_t i= 0;
    for(; i + 16 <= size; i+= 16){
        _mm_prefetch(reinterpret_cast<const char*>(&v[i+ 64]), _MM_HINT_T0);
        __m256 vec1_fp= _mm256_load_ps(&v[i]);
        __m256 vec2_fp= _mm256_load_ps(&v[i + 8]);

        vec1_fp= clamp(vec1_fp, MM256_MINQ, MM256_MAXQ);
        vec2_fp= clamp(vec2_fp, MM256_MINQ, MM256_MAXQ);
        
        __m256 vec1_fp_scaled= _mm256_fmadd_ps(vec1_fp, MM256_SCALE, MM256_ROUND);
        __m256 vec2_fp_scaled= _mm256_fmadd_ps(vec2_fp, MM256_SCALE, MM256_ROUND);

        __m256i vec1_pi= _mm256_cvtps_epi32(vec1_fp_scaled);
        __m256i vec2_pi= _mm256_cvtps_epi32(vec2_fp_scaled);

        __m256i vec_pi= _mm256_packs_epi32(vec1_pi, vec2_pi);
        _mm256_store_si256(reinterpret_cast<__m256i*>(&q[i]), vec_pi);
    }

    for (; i < size; i++) {
        float val= v[i];
        val= clamp(val, MINQ, MAXQ);
        val= val * SCALE_FACTOR + ROUND_FACTOR;
        q[i]= static_cast<int16_t>(val);
    }
}

int32_t dotproduct_q8_8(int16_t* w_q8_8, int16_t* x_q8_8, size_t size){
    __m256i vec_sum_q16_16 = _mm256_setzero_si256();
    size_t i = 0;
    for(; i + 32 <= size; i += 32){ 
        _mm_prefetch(reinterpret_cast<const char*>(&w_q8_8[i+ 64]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&x_q8_8[i+ 64]), _MM_HINT_T0);

        __m256i vec1_w_q8_8 = _mm256_load_si256((__m256i*)&w_q8_8[i]);
        __m256i vec1_x_q8_8 = _mm256_load_si256((__m256i*)&x_q8_8[i]);
        __m256i vec2_w_q8_8 = _mm256_load_si256((__m256i*)&w_q8_8[i+ 16]);
        __m256i vec2_x_q8_8 = _mm256_load_si256((__m256i*)&x_q8_8[i+ 16]);

        __m256i dot1= _mm256_madd_epi16(vec1_w_q8_8, vec1_x_q8_8);
        __m256i dot2= _mm256_madd_epi16(vec2_w_q8_8, vec2_x_q8_8);

        __m256i prod= _mm256_add_epi32(dot1, dot2);
        vec_sum_q16_16 = _mm256_add_epi32(vec_sum_q16_16, prod);
    }

    __m128i sum_q16_16_lower= _mm256_castsi256_si128(vec_sum_q16_16);
    __m128i sum_q16_16_higher= _mm256_extracti128_si256(vec_sum_q16_16, 1);
    __m128i sum_q16_16_128= _mm_add_epi32(sum_q16_16_lower, sum_q16_16_higher);

    sum_q16_16_128= _mm_hadd_epi32(sum_q16_16_128, sum_q16_16_128);
    sum_q16_16_128= _mm_hadd_epi32(sum_q16_16_128, sum_q16_16_128);
    int32_t sum_q16_16= _mm_cvtsi128_si32(sum_q16_16_128);

    for(; i < size; i++){
        sum_q16_16 += w_q8_8[i] * x_q8_8[i];
    }

    return sum_q16_16;
}

float dotproduct_fp(float* w_fp, float* x_fp, size_t size){
    __m256 vec_sum_fp= _mm256_setzero_ps();
    size_t i = 0;
    for(; i + 16<= size; i += 16){ 
        _mm_prefetch(reinterpret_cast<const char*>(&w_fp[i+ 32]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&x_fp[i+ 32]), _MM_HINT_T0);

        __m256 vec1_w_fp = _mm256_load_ps(&w_fp[i]);
        __m256 vec1_x_fp = _mm256_load_ps(&x_fp[i]);
        __m256 vec2_w_fp = _mm256_load_ps(&w_fp[i+ 8]);
        __m256 vec2_x_fp = _mm256_load_ps(&x_fp[i+ 8]);

        __m256 dot1= _mm256_fmadd_ps(vec1_w_fp, vec1_x_fp, _mm256_setzero_ps());
        __m256 dot2= _mm256_fmadd_ps(vec2_w_fp, vec2_x_fp, _mm256_setzero_ps());

        __m256 prod= _mm256_add_ps(dot1, dot2);
        vec_sum_fp= _mm256_add_ps(vec_sum_fp, prod);
    }

    __m128 sum_fp_lower= _mm256_castps256_ps128(vec_sum_fp);
    __m128 sum_fp_higher= _mm256_extractf128_ps(vec_sum_fp, 1);
    __m128 sum_fp_128= _mm_add_ps(sum_fp_lower, sum_fp_higher); //indices: [1, 2, 3, 4] 

    sum_fp_128= _mm_hadd_ps(sum_fp_128, sum_fp_128); //[1+2, 3+4, 1+2. 3+4]
    sum_fp_128= _mm_hadd_ps(sum_fp_128, sum_fp_128); //[1+2+3+4, ....]
    float sum_fp= _mm_cvtss_f32(sum_fp_128);

    for(; i < size; i++){
        sum_fp += w_fp[i] * x_fp[i];
    }
    return sum_fp;
}

//delta= lr * (y_hat - y)x^T
void sgd_inplace(int16_t y_hat, float y, float* w_fp, float* x_fp, size_t size, float lr){
    float neg_coeff= lr*(y- q8_8_to_float(y_hat));
    __m256 vec_neg_coeff= _mm256_broadcast_ss(&neg_coeff); 

    size_t i= 0;    
    for (; i < size ; i += 16){

        _mm_prefetch(reinterpret_cast<const char*>(&x_fp[i + 32]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&w_fp[i + 32]), _MM_HINT_T0);
        
        __m256 vec1_x_fp= _mm256_load_ps(&x_fp[i]);
        __m256 vec1_w_fp= _mm256_load_ps(&w_fp[i]);
        __m256 vec2_x_fp= _mm256_load_ps(&x_fp[i+8]);
        __m256 vec2_w_fp= _mm256_load_ps(&w_fp[i+8]);

        __m256 vec1_w_fp_new=  _mm256_fmadd_ps(vec_neg_coeff, vec1_x_fp, vec1_w_fp);
        __m256 vec2_w_fp_new=  _mm256_fmadd_ps(vec_neg_coeff, vec2_x_fp, vec2_w_fp);

        _mm256_store_ps(&w_fp[i], vec1_w_fp_new);
        _mm256_store_ps(&w_fp[i + 8], vec2_w_fp_new);
    }

    for (; i < size; i ++){
        w_fp[i]+= neg_coeff*x_fp[i];
    }
}

//refer to pseudocode
void adamW_inplace(int16_t y_hat, float y, float* w_fp, float* x_fp, size_t size, AdamWParams& parmas){
    return;
}

