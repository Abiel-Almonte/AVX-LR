#pragma once

#include <stdlib.h>
#include <cstring>
#include <stdexcept>
#include <cassert>

template <typename T> 
class alignedArray{
    private: 
        size_t size_m;
        T* ptr_m;

    public:
        explicit alignedArray(size_t size): size_m(size), ptr_m(nullptr){
            ptr_m= static_cast<T*>(std::aligned_alloc(32, size*sizeof(T)));
            if(!ptr_m){
                throw std::bad_alloc();
            }
        }
        
        alignedArray(size_t alignment, size_t size): size_m(size), ptr_m(nullptr){
            ptr_m= static_cast<T*>(std::aligned_alloc(alignment, size*sizeof(T)));
            if (!ptr_m){
                throw std::bad_alloc();
            }
        }

        alignedArray(): size_m(0), ptr_m(nullptr){};

        alignedArray(const alignedArray&)= delete;
        alignedArray& operator=(const alignedArray&)= delete;

        alignedArray(alignedArray&& other) noexcept: size_m(other.size_m), ptr_m(other.ptr_m){
            other.ptr_m= nullptr;
            other.size_m= 0;
        }

        alignedArray& operator=(alignedArray&& other) noexcept {
            if (this != &other){
                if (ptr_m){
                    free(ptr_m);
                }
                ptr_m= other.ptr_m;
                size_m= other.size_m;
                other.ptr_m= nullptr;
                other.size_m= 0;
            }
            return *this;
        }

        alignedArray<T> deepCopy() const{
            alignedArray<T> copy(size_m);
            std::memcpy(copy.data(), ptr_m, size_m* sizeof(T));
            return copy;
        }        

        ~alignedArray(){
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
    const float learning_rate;
    const float beta1;
    const float beta2;
    const float beta1_complement;
    const float beta2_complement;
    float beta1i;
    float beta2i;
    const float decay;
    const float eps;

    alignedArray<float> moment1;
    alignedArray<float> moment2;

    AdamWParams(size_t size, float lr = 0.001);

    private:
        AdamWParams();
        
};
