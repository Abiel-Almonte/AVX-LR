#include <immintrin.h>
#include <array>
#include <cmath>
#include <iostream>

constexpr float MAX_Q= 127.9940f;
constexpr float MIN_Q= -128.0f;
constexpr float SCALE_FACTOR= 256.0f;
constexpr float ROUND_FACTOR= 0.5f;

constexpr inline int16_t float_to_q8_8(float n){
    return static_cast<int16_t>(n* SCALE_FACTOR + ROUND_FACTOR);
}

constexpr inline float q8_8_to_float(int16_t n){
    return static_cast<float>(n)/ SCALE_FACTOR; 
}

constexpr inline float sigmoid(float x){
    return 1.0f/ (1.0f + std::exp(-x));
}

constexpr std::array<int16_t, 2048> precompute_sigmoid(){
    std::array<int16_t, 2048> sigmoidTable{};
    sigmoidTable[0]= float_to_q8_8(0); 
    sigmoidTable[2047]= float_to_q8_8(1);
    for(int i= 1; i < 2047; i++){
        float input_float= q8_8_to_float(i - 1024);
        sigmoidTable[i]=  float_to_q8_8(sigmoid(input_float));
    }
    return sigmoidTable;
}

//CXX 20
alignas(64) constexpr std::array<int16_t, 2048> SIGMOID_TABLE = precompute_sigmoid();

const __m256 MAX_QUANT= _mm256_set1_ps(MAX_Q);
const __m256 MIN_QUANT= _mm256_set1_ps(MIN_Q);
const __m256 SCALE= _mm256_set1_ps(SCALE_FACTOR);
const __m256 ROUND= _mm256_set1_ps(ROUND_FACTOR);

static inline __m256 clamp(__m256 n, __m256 min, __m256 max){
    return _mm256_max_ps(min, _mm256_min_ps(max, n));
}

static inline float clamp(float n, float min, float max){
    return  std::max(min, std::min(max, n));
}

static inline int16_t clamp(int16_t n, int16_t min, int16_t max){
    return  std::max(min, std::min(max, n));
}

int16_t sigmoidApprox_q8_8(int16_t n){
    int16_t i= clamp(n, -1024, 1023) + 1024;
    return SIGMOID_TABLE[i];
}

int16_t sigmoidApprox_q8_8(int32_t n){
    int16_t n_int16= static_cast<int16_t>(n >>8);
    int16_t i= clamp(n_int16, -1024, 1023) + 1024;
    return SIGMOID_TABLE[i];
}

float sigmoidApprox_float(int16_t n){
    return q8_8_to_float(sigmoidApprox_q8_8(n));
}

float sigmoidApprox_float(int32_t n){
    return q8_8_to_float(sigmoidApprox_q8_8(n));
}

void quantize8_8_inplace(float* v, int16_t* q, int size){
    //Do not use size where size%32 != 0. Or feel latency
    int i= 0;
    for(; i <= size - 16; i+= 16){
        _mm_prefetch(reinterpret_cast<const char*>(&v[i+ 64]), _MM_HINT_T0);
        __m256 vec1_fp= _mm256_load_ps(&v[i]);
        __m256 vec2_fp= _mm256_load_ps(&v[i + 8]);

        vec1_fp= clamp(vec1_fp, MIN_QUANT, MAX_QUANT);
        vec2_fp= clamp(vec2_fp, MIN_QUANT, MAX_QUANT);
        
        __m256 vec1_fp_scaled= _mm256_fmadd_ps(vec1_fp, SCALE, ROUND);
        __m256 vec2_fp_scaled= _mm256_fmadd_ps(vec2_fp, SCALE, ROUND);

        __m256i vec1_pi= _mm256_cvtps_epi32(vec1_fp_scaled);
        __m256i vec2_pi= _mm256_cvtps_epi32(vec2_fp_scaled);

        __m256i vec_pi= _mm256_packs_epi32(vec1_pi, vec2_pi);
        _mm256_store_si256(reinterpret_cast<__m256i*>(&q[i]), vec_pi);
    }

    for (; i < size; i++) {
        float val= v[i];
        val= clamp(val, MIN_Q, MAX_Q);
        val= val * SCALE_FACTOR + ROUND_FACTOR;
        q[i]= static_cast<int16_t>(val);
    }
}
int16_t dotproduct_q8_8(int16_t* w_q8_8, int16_t* x_q8_8, int size){
    __m256i vec_sum_q16_16 = _mm256_setzero_si256();
    int i = 0;
    for(; i <= size - 32; i += 32){ 
        _mm_prefetch(reinterpret_cast<const char*>(&w_q8_8[i+ 128]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&x_q8_8[i+ 128]), _MM_HINT_T0);

        __m256i vec1_w_q8_8 = _mm256_load_si256((__m256i*)&w_q8_8[i]);
        __m256i vec1_x_q8_8 = _mm256_load_si256((__m256i*)&x_q8_8[i]);
        __m256i dot1= _mm256_madd_epi16(vec1_w_q8_8, vec1_x_q8_8);

        __m256i vec2_w_q8_8 = _mm256_load_si256((__m256i*)&w_q8_8[i+ 16]);
        __m256i vec2_x_q8_8 = _mm256_load_si256((__m256i*)&x_q8_8[i+ 16]);
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
    return static_cast<int16_t>(sum_q16_16 >> 8);
}

template <typename T> 
class alignedVector{
    private: 
        size_t size_m;
        T* ptr_m;

    public:
        explicit alignedVector(size_t size): size_m(size), ptr_m(nullptr){
            ptr_m= static_cast<T*>(std::aligned_alloc(32, size*sizeof(T)));
            if(!ptr_m){
                throw std::bad_alloc();
            }
        }
        
        alignedVector(size_t alignment, size_t size): size_m(size), ptr_m(nullptr){
            ptr_m= static_cast<T*>(std::aligned_alloc(alignment, size*sizeof(T)));
            if (!ptr_m){
                throw std::bad_alloc();
            }
        }

        alignedVector(): size_m(0), ptr_m(nullptr){};

        alignedVector(const alignedVector&)= delete;
        alignedVector& operator=(const alignedVector&)= delete;

        alignedVector(alignedVector&& other) noexcept: size_m(other.size_m), ptr_m(other.ptr_m){
            other.ptr_m= nullptr;
            other.size_m= 0;
        }

        alignedVector& operator=(alignedVector&& other) noexcept {
            if (this != &other){
                free(ptr_m);
                ptr_m= other.ptr_m;
                size_m= other.size_m;
                other.ptr_m= nullptr;
                other.size_m= 0;
            }
            return *this;
        }

        ~alignedVector(){
            free(ptr_m);
        }

        T* data() const {
            return ptr_m;
        }
        
        size_t size() const {
            return size_m;
        }

        const T& operator [] (size_t index) const {
            assert(index < size_m && "Index out of bounds");
            return *(ptr_m + index);
        }

        T& operator [] (size_t index) {
            assert(index < size_m && "Index out of bounds");
            return *(ptr_m + index);
        }

};

struct AdamWParams{
    const float beta1;
    const float beta2;
    const float beta1_complement;
    const float beta2_complement;
    float beta1i;
    float beta2i;
    const float decay;
    const float learning_rate;
    const float eps;

    alignedVector<float> moment1;
    alignedVector<float> moment2;

    AdamWParams(size_t size): 
        learning_rate(1e-3f), 
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
        
};

#include <random>
#include <chrono>
#include <algorithm>

float dotproduct_scalar_fp(float* weights, float* activations, int size) {
    float accumulated = 0.0f;
    #pragma clang loop vectorize(disable)
    for (int i = 0; i < size; i++) {
        accumulated += weights[i] * activations[i];
    }
    return accumulated;
}

int main() {
    return 0;
}