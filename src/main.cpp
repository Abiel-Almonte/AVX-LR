#include "utils/containers.hh"
#include "utils/avx.hh"
#include "utils/tools.hh"
#include "utils/scalar.hh"

#include <iostream>
#include <random>
#include <chrono>

void analyze_timings(std::vector<double> timings, const char* title){
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

void analyze_errors(std::vector<double> errors, const char* title) {
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
    std::cout << "Median Relative Error  (P50): " << median * 100 << " %" << std::endl;
    std::cout << "Mean Relative Error    (Avg): " << mean * 100 << " %" << std::endl;
    std::cout << "95th Percentile Relative Error (P95): " << p95 * 100 << " %" << std::endl;
    std::cout << "99th Percentile Relative Error (P99): " << p99 * 100 << " %" << std::endl;
}


template <size_t feature_size>
void benchmark_inference(int iterations){
    alignedArray<int16_t> avx_q8_8_weights(feature_size);
    alignedArray<int16_t> avx_q8_8_inputs(feature_size);

    alignedArray<float> avx_weights(feature_size);
    alignedArray<float> avx_inputs(feature_size);

    std::array<float, feature_size> scalar_weights{};
    std::array<float, feature_size> scalar_inputs{};
    
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist_weights(-128, 128);
    std::uniform_real_distribution<float> dist_inputs(-1, 1);

    for (size_t i = 0; i< feature_size; i++){
        float rand_w= dist_weights(mt);
        float rand_x= dist_inputs(mt);

        scalar_weights[i]= rand_w;
        scalar_inputs[i]= rand_x;
        
        avx_weights[i]= rand_w;
        avx_inputs[i]= rand_x;
    }

    quantize8_8_inplace(scalar_weights.data(), avx_q8_8_weights.data(), feature_size);
    quantize8_8_inplace(scalar_inputs.data(), avx_q8_8_inputs.data(), feature_size);

    std::vector<double> avx_q8_8_latency{};
    std::vector<double> avx_latency{};
    std::vector<double> scalar_latency{};
    std::vector<double> relative_errors{};
    scalar_latency.reserve(iterations);
    avx_latency.reserve(iterations);
    avx_q8_8_latency.reserve(iterations);

    volatile float accumulation= 0.0f;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> duration;

    for(int i= 0; i < iterations; i++){ 
        
        start= std::chrono::high_resolution_clock::now();
        float scalar_output= sigmoid_scalar(dotproduct_scalar(scalar_weights.data(), scalar_inputs.data(), feature_size));
        end= std::chrono::high_resolution_clock::now();
        duration= end - start;
        scalar_latency.push_back(duration.count()* 1e9);

        start= std::chrono::high_resolution_clock::now();
        int16_t avx_output_q8_8= sigmoidApprox_fp_to_q8_8(dotproduct_fp(avx_weights.data(), avx_inputs.data(), feature_size));
        end= std::chrono::high_resolution_clock::now();
        duration= end -start;
        avx_latency.push_back(duration.count()* 1e9);
        accumulation+= q8_8_to_float(avx_output_q8_8);
        
        start= std::chrono::high_resolution_clock::now();
        int16_t avx_q8_8_output= sigmoidApprox_q8_8(dotproduct_q8_8(avx_q8_8_weights.data(), avx_q8_8_inputs.data(), feature_size));
        end= std::chrono::high_resolution_clock::now();
        duration= end -start;
        avx_q8_8_latency.push_back(duration.count()* 1e9);

        float avx_q8_8_output_fp= q8_8_to_float(avx_q8_8_output);

        double rel_error = (std::fabs(scalar_output) > 1e-6) ? std::fabs(avx_q8_8_output_fp - scalar_output) / std::fabs(scalar_output) : std::fabs(avx_q8_8_output_fp - scalar_output);
        relative_errors.push_back(rel_error);

        accumulation+= avx_q8_8_output_fp;
        accumulation+= scalar_output;
    }

    analyze_timings(scalar_latency, "Scalar Inference");
    analyze_timings(avx_latency, "AVX Inferece");
    analyze_timings(avx_q8_8_latency, "AVX Q(8.8) Inference");
    analyze_errors(relative_errors, "AVX Q(8.8) Error");

}


int main() {
    benchmark_inference<32>(1e9);
    benchmark_inference<8192>(1e9);
    return 0;
}