#include "utils/containers.hh"
#include "utils/avx.hh"
#include "utils/tools.hh"
#include "utils/scalar.hh"

#include <iostream>
#include <random>
#include <chrono>

void analyze_timings(std::vector<double>& timings, const char* title){
     if (timings.empty()) {
        std::cout << "No data provided for " << title << std::endl;
        return;
    }
    
    std::sort(timings.begin(), timings.end());
    double median= timings[timings.size()/2];
    double mean= std::accumulate(timings.begin(), timings.end(), 0.0)/timings.size(); 
    double p95= timings[static_cast<int>(timings.size()*0.95)];
    double p99= timings[static_cast<int>(timings.size()*0.99)];

    std::cout << "===== "<< title << " Speed Benchmark =====" << std::endl;
    std::cout << "Median Latency  (P50): " << median << " ns" << std::endl;
    std::cout << "Mean Latency    (Avg): " << mean << " ns" << std::endl;
    std::cout << "95th Percentile (P95): " << p95 << " ns" << std::endl;
    std::cout << "99th Percentile (P99): " << p99 << " ns" << std::endl;
}

void analyze_errors(std::vector<double>& errors, const char* title) {
    if (errors.empty()) {
        std::cout << "No error data provided for " << title << std::endl;
        return;
    }
    
    std::sort(errors.begin(), errors.end());
    double median = errors[errors.size() / 2];
    double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    double p95 = errors[static_cast<int>(errors.size() * 0.95)];
    double p99 = errors[static_cast<int>(errors.size() * 0.99)];

    std::cout << "===== " << title << " Error Benchmark =====" << std::endl;
    std::cout << "Median Absolute Error (P50): " << median << std::endl;
    std::cout << "Mean Absolute Error   (Avg): " << mean << std::endl;
    std::cout << "95th Percentile Absolute Error (P95): " << p95 << std::endl;
    std::cout << "99th Percentile Absolute Error (P99): " << p99 << std::endl;
}

void analyze_p95_speedup(std::vector<double>& timings, std::vector<double>& relativeTo, const char* title){
    if (timings.empty() || relativeTo.empty()) {
        std::cout << "No timing data provided for " << title << std::endl;
        return;
    }

    std::sort(timings.begin(), timings.end());
    std::sort(relativeTo.begin(), relativeTo.end());
    double timings_p95= timings[static_cast<int>(timings.size()*0.95)];
    double relativeTo_p95= relativeTo[static_cast<int>(relativeTo.size()* 0.95)];
    double speedup= (relativeTo_p95/timings_p95);

    std::cout << "===== " << title << " P95 Speed Comparision =====" << std::endl;
    double multiplier= std::pow(10, 2);
    if (speedup < 1){ 
        std::cout <<  std::round(1/speedup * multiplier)/multiplier << "x slowdown" << std::endl; 
    }
    else{
        std::cout << std::round(speedup* multiplier)/ multiplier << "x speedup" << std::endl;
    }

}

template <size_t feature_size>
void benchmark_inference(int iterations, int reps) {
    alignedArray<int16_t> avx_q8_8_weights(feature_size);
    alignedArray<int16_t> avx_q8_8_inputs(feature_size);

    alignedArray<float> avx_weights(feature_size);
    alignedArray<float> avx_inputs(feature_size);

    std::array<float, feature_size> scalar_weights{};
    std::array<float, feature_size> scalar_inputs{};
    
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist_weights(-1, 1);
    std::uniform_real_distribution<float> dist_inputs(-1, 1);

    std::vector<double> avx_q8_8_latency{};
    std::vector<double> avx_latency{};
    std::vector<double> scalar_latency{};
    std::vector<double> absolute_errors_q8_8{};
    std::vector<double> absolute_errors_fp{};
    scalar_latency.reserve(iterations);
    avx_latency.reserve(iterations);
    avx_q8_8_latency.reserve(iterations);
    absolute_errors_q8_8.reserve(iterations);
    absolute_errors_fp.reserve(iterations);

    volatile float accumulation = 0.0f;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    for (int w = 0; w < 100; w++) {
        for (size_t j = 0; j < feature_size; j++) {
            float rw= dist_weights(mt);
            float rx= dist_inputs(mt);
            scalar_weights[j]= rw;
            scalar_inputs[j]= rx;
            avx_weights.data()[j]= rw;
            avx_inputs.data()[j]= rx;
        }
        quantize8_8_inplace(scalar_weights.data(), avx_q8_8_weights.data(), feature_size);
        quantize8_8_inplace(scalar_inputs.data(),  avx_q8_8_inputs.data(),  feature_size);
        accumulation+= sigmoid_fp(dotproduct_scalar(scalar_weights.data(), scalar_inputs.data(), feature_size));
        accumulation+= q8_8_to_float(sigmoidApprox_fp_to_q8_8(dotproduct_fp(avx_weights.data(), avx_inputs.data(), feature_size)));
        accumulation+= q8_8_to_float(sigmoidApprox_q8_8(dotproduct_q8_8(avx_q8_8_weights.data(), avx_q8_8_inputs.data(), feature_size)));
    }

    for (int iter = 0; iter < iterations; iter++) {
        for (size_t j = 0; j < feature_size; j++) {
            float rand_w= dist_weights(mt);
            float rand_x= dist_inputs(mt);
            scalar_weights[j]= rand_w;
            scalar_inputs[j]= rand_x;
            avx_weights.data()[j]= rand_w;
            avx_inputs.data()[j]= rand_x;
        }

        quantize8_8_inplace(scalar_weights.data(), avx_q8_8_weights.data(), feature_size);
        quantize8_8_inplace(scalar_inputs.data(),  avx_q8_8_inputs.data(),  feature_size);

        auto start= std::chrono::high_resolution_clock::now();
        float scalar_result = 0.0f;
        for (int r= 0; r < reps; r++) {
            scalar_result += sigmoid_fp(dotproduct_scalar(scalar_weights.data(), scalar_inputs.data(), feature_size));
        }
        auto end= std::chrono::high_resolution_clock::now();
        double scalar_time= std::chrono::duration<double>(end - start).count() / reps;
        scalar_latency.push_back(scalar_time * 1e9);
        accumulation+= scalar_result;

        start = std::chrono::high_resolution_clock::now();
        int16_t avx_result_q8_8 = 0;
        for (int r= 0; r < reps; r++) {
            avx_result_q8_8+= sigmoidApprox_fp_to_q8_8(dotproduct_fp(avx_weights.data(), avx_inputs.data(), feature_size));
        }
        end = std::chrono::high_resolution_clock::now();
        double avx_time= std::chrono::duration<double>(end - start).count() / reps;
        avx_latency.push_back(avx_time * 1e9);
        float avx_result_fp= q8_8_to_float(avx_result_q8_8);
        absolute_errors_fp.push_back(std::fabs((avx_result_fp - scalar_result) / reps));
        accumulation += avx_result_fp;

        start = std::chrono::high_resolution_clock::now();
        int16_t avx_q8_8_result= 0;
        for (int r= 0; r < reps; r++) {
            avx_q8_8_result+= sigmoidApprox_q16_16_to_q8_8(dotproduct_q8_8(avx_q8_8_weights.data(), avx_q8_8_inputs.data(), feature_size));
        }
        end= std::chrono::high_resolution_clock::now();
        double avx_q8_8_time= std::chrono::duration<double>(end - start).count() / reps;
        avx_q8_8_latency.push_back(avx_q8_8_time * 1e9);
        float avx_q8_8_result_fp= q8_8_to_float(avx_q8_8_result);
        absolute_errors_q8_8.push_back(std::fabs((avx_q8_8_result_fp - scalar_result) / reps));
        accumulation+= avx_q8_8_result_fp;
    }

    std::cout << "===== " << feature_size << " Features =====" << std::endl;
    std::cout << std::endl;
    analyze_timings(scalar_latency, "Scalar FP32 Inference");
    analyze_timings(avx_latency, "AVX FP32 Inference");
    analyze_timings(avx_q8_8_latency, "AVX Q(8.8) Inference");
    std::cout << std::endl;
    analyze_errors(absolute_errors_q8_8, "AVX Q(8.8) vs Scalar");
    analyze_errors(absolute_errors_fp, "AVX FP32 vs Scalar");
    std::cout << std::endl;
    analyze_p95_speedup(avx_q8_8_latency, scalar_latency, "AVX Q(8.8) vs Scalar");
    analyze_p95_speedup(avx_latency, scalar_latency, "AVX FP32 vs Scalar");
    analyze_p95_speedup(avx_q8_8_latency, avx_latency, "AVX Q(8.8) vs AVX FP32");
    std::cout << std::endl;
    std::cout << "Accumulation (to avoid optimization): " << accumulation << std::endl;
}


int main() {

    benchmark_inference<32>(1e6, 100);
    benchmark_inference<64>(1e6, 100);
    benchmark_inference<1024>(1e6, 100);
    benchmark_inference<4096>(1e6, 100);
    benchmark_inference<8192>(1e6, 100);
    benchmark_inference<16384>(1e6, 100);
    benchmark_inference<32768>(1e6, 100);
    
    return 0;
}