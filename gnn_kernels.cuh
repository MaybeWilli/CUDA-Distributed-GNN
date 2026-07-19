#ifndef GNN_KERNELS_H
#define GNN_KERNELS_H

#include <stdio.h>
#include <assert.h>
#include <random>

constexpr int FEATURE_DIM = 16;

inline
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

namespace GnnKernels
{
    __global__ void forward_run(float* features, int* edges, int* offsets, float* output_layer, 
        float* probabilities, float* layer_weights, float* projection, int nodes);

    __global__ void prob_backprop(int* edges, int* offsets, float* output_layer, float* probabilities, int* weights,
        float* layer_weights, float* projection, float* node_gradients, int nodes);

    __global__ void score_backprop(float* probabilities, float* node_gradients, int nodes);

    __global__ void proj_backprop(float* output_layer, float* node_gradients, float* proj_gradients, int nodes);

    __global__ void feature_backprop(float* feature_gradients, float* node_gradients, float* projection, int nodes);

    __global__ void tanh_backprop(float* feature_gradients, float* output_layer, float* projection, int nodes);

    __global__ void weight_backprop(int* edges, int* offsets, float* features, float* feature_gradients, float* layer_weight_gradients, int nodes);

    __global__ void update_weights(float* layer_weight_gradients, float* layer_weights, float* projection_gradients, 
        float* projection, float lr, int nodes);

    __global__ void request_probabilities(float* probabilities, int offset, int nodes, int* requests, int* request_locations, 
        float* buffer, int buffer_offset, int request_size);
    
    __global__ void fill_nan(float* buffer, int size);

    __global__ void update_probabilities(float* probabilities, float* buffer, int nodes, int offset);
}

#endif