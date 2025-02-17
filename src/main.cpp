#include "utils/containers.hh"
#include "utils/avx.hh"
#include "utils/tools.hh"
#include "utils/scalar.hh"

#include <iostream>
#include <random>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

json analyze_timings(std::vector<double>& timings, const std::string& title){
    json result; 
    if (timings.empty()) {
        result["error"]= "No data provided for " + title;
        return result;
    }
    
    std::sort(timings.begin(), timings.end());
    double median= timings[timings.size()/2];
    double mean= std::accumulate(timings.begin(), timings.end(), 0.0)/timings.size(); 
    double p95= timings[static_cast<int>(timings.size()*0.95)];
    double p99= timings[static_cast<int>(timings.size()*0.99)];

    result["title"]= title;
    result["median_ns"]= median;
    result["mean_ns"]= mean;
    result["p95_ns"]= p95;
    result["p99_ns"]= p99;

    return result;
}

json analyze_errors(std::vector<double>& errors, const std::string& title) {
    json result;
    if (errors.empty()) {
        result["error"]= "No data provided for " + title;
        return result;
    }
    
    std::sort(errors.begin(), errors.end());
    double median = errors[errors.size() / 2];
    double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    double p95 = errors[static_cast<int>(errors.size() * 0.95)];
    double p99 = errors[static_cast<int>(errors.size() * 0.99)];

    result["title"]= title;
    result["median_error"]= median;
    result["mean_error"]= mean;
    result["p95_error"]= p95;
    result["p99_error"]= p99;

    return result;
}

json analyze_p95_speedup(std::vector<double>& timings, std::vector<double>& relativeTo, const std::string& title){
    json result;
    if (timings.empty() || relativeTo.empty()) {
        result["error"]= "No data provided for " + title;
        return result;
    }

    std::sort(timings.begin(), timings.end());
    std::sort(relativeTo.begin(), relativeTo.end());
    double timings_p95= timings[static_cast<int>(timings.size()*0.95)];
    double relativeTo_p95= relativeTo[static_cast<int>(relativeTo.size()* 0.95)];
    double speedup= (relativeTo_p95/timings_p95);

    result["title"]= title;
    double multiplier= std::pow(10, 2);
    if (speedup < 1){ 
        result["x slowdown"]= std::round(1/speedup * multiplier)/multiplier;
    }
    else{
        result["x speedup"]=  std::round(speedup* multiplier)/ multiplier;
    }

    return result;
}

template <size_t feature_size>
json benchmark_inference(int iterations, int reps) {
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
            avx_q8_8_result+= sigmoidApprox_q8_8(dotproduct_q8_8(avx_q8_8_weights.data(), avx_q8_8_inputs.data(), feature_size));
        }
        end= std::chrono::high_resolution_clock::now();
        double avx_q8_8_time= std::chrono::duration<double>(end - start).count() / reps;
        avx_q8_8_latency.push_back(avx_q8_8_time * 1e9);
        float avx_q8_8_result_fp= q8_8_to_float(avx_q8_8_result);
        absolute_errors_q8_8.push_back(std::fabs((avx_q8_8_result_fp - scalar_result) / reps));
        accumulation+= avx_q8_8_result_fp;
    }

    json benchmark_results;
    benchmark_results["Scalar_FP32_latency"]= analyze_timings(scalar_latency, "Scalar FP32 Inference");
    benchmark_results["AVX_FP32_latency"]= analyze_timings(avx_latency, "AVX FP32 Inference");
    benchmark_results["AVX_Q88_latency"]= analyze_timings(avx_q8_8_latency, "AVX Q(8.8) Inference");
    benchmark_results["AVX_Q88_Error"]= analyze_errors(absolute_errors_q8_8, "AVX Q(8.8) vs Scalar");
    benchmark_results["AVX_FP32_Error"]= analyze_errors(absolute_errors_fp, "AVX FP32 vs Scalar");
    benchmark_results["AVX_Q88_Scalar_Speedup"]= analyze_p95_speedup(avx_q8_8_latency, scalar_latency, "AVX Q(8.8) vs Scalar");
    benchmark_results["AVX_FP32_Scalar_Speedup"]= analyze_p95_speedup(avx_latency, scalar_latency, "AVX FP32 vs Scalar");
    benchmark_results["AVX_Q88_Fp32_Speedup"]= analyze_p95_speedup(avx_q8_8_latency, avx_latency, "AVX Q(8.8) vs AVX FP32");
    std::cout << "Accumulation (to avoid optimization): " << accumulation << std::endl;

    return benchmark_results;
}

#include <fstream>

int main() {
    std::ofstream file("data.json");
    json data;

    data["32"]= benchmark_inference<32>(1e6, 100);
    data["64"]= benchmark_inference<64>(1e6, 100);
    data["1024"]= benchmark_inference<1024>(1e6, 100);
    data["4096"]= benchmark_inference<4096>(1e6, 100);
    data["8192"]= benchmark_inference<8192>(1e6, 100);
    data["16384"]= benchmark_inference<16384>(1e6, 100);
    data["32768"]= benchmark_inference<32768>(1e6, 100);
    
    file << data.dump(4);
    file.close();
    return 0;
}