#include "avx.hh"
#include "containers.hh"

class SGDLogisicRegression{
    private:
        const size_t feature_size_m;
        alignedArray<float> weights_m;
        
        float learning_rate;
        int16_t threshold;
        
    public:
        
        SGDLogisicRegression(size_t feature_size);
        SGDLogisicRegression(size_t feature_size, float learning_rate);
        SGDLogisicRegression(size_t feature_size, float learning_rate, float threshold);
        SGDLogisicRegression(size_t alignment, size_t feature_size, float learing_rate, float threshold);
        
        void setThreshold(float val);
        void setLearningRate(float val);
        
        int16_t inference(alignedArray<float>& inputs);
        void update(int16_t prediction, float label);
};