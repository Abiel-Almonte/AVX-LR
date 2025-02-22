#include "avx.hh"
#include "containers.hh"

class SGDLogisticRegression{
    private:
        const size_t feature_size_m;
        
        float learning_rate_m;
        float threshold_m;
        
        alignedArray<float> weights_m;
        alignedArray<float> inputs_m;
        alignedArray<int16_t> weights_q8_8_m;
        alignedArray<int16_t> inputs_q8_8_m;

        int16_t inference_q8_8(alignedArray<float>& inputs);
        void initWeights();
        
    public:
        
        SGDLogisticRegression(size_t feature_size, float learning_rate= 0.01f, float threshold= 0.0f, size_t alignment= 32);
        
        void setThreshold(float val);
        void setLearningRate(float val);
        void setInputs(float* x);
        
        float inference_fp(alignedArray<float>& inputs);
        float inference_q8_8_to_fp(alignedArray<float>& inputs);
        void update_weights(float prediction, float label);
};