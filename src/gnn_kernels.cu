#include <stdio.h>
#include <assert.h>
#include <random>
#include "gnn_kernels.cuh"

__device__ __forceinline__
float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__global__ void GnnKernels::forward_run(float* features, int* edges, int* offsets, float* output_layer, 
    float* probabilities, float* layer_weights, float* projection, int nodes)
{
    int warp = threadIdx.x / 32 + blockIdx.x * (blockDim.x / 32);
    int lane = threadIdx.x % 32;
    int stride = gridDim.x * (blockDim.x / 32);

    for (int v = warp; v < nodes; v += stride)
    {
        float output[FEATURE_DIM];
        for (int i = 0; i < FEATURE_DIM; i++)
        {
            output[i] = 0;
        }
        for (int edge = offsets[v]+lane; edge < offsets[v+1]; edge += 32)
        {
            int u = edges[edge];
            for (int i = 0; i < FEATURE_DIM; i++)
            {
                output[i] += features[u * FEATURE_DIM + i];
            }
        }
        for (int delta = 16; delta > 0; delta >>= 1)
        {
            for (int i = 0; i < FEATURE_DIM; i++)
            {
                output[i] += __shfl_down_sync(0xffffffff, output[i], delta);
            }
        }

        if (lane == 0)
        {
            float score = 0;
            float temp[FEATURE_DIM];
            for (int i = 0; i < FEATURE_DIM; i++)
            {
                temp[i] = 0;
            }
            for (int j = 0; j < FEATURE_DIM; j++)
            {
                for (int k = 0; k < FEATURE_DIM; k++)
                {
                    temp[j] += output[k] * layer_weights[j * FEATURE_DIM + k];
                }
            }
            for (int i = 0; i < FEATURE_DIM; i++)
            {
                temp[i] = tanhf(temp[i]);
                output_layer[v * FEATURE_DIM + i] = temp[i];
                score += temp[i] * projection[i];
            }
            probabilities[v] = sigmoid(score);
        }
    }

}

__global__ void GnnKernels::prob_backprop(int* edges, int* offsets, float* output_layer, float* probabilities, int* weights,
    float* layer_weights, float* projection, float* node_gradients, int nodes)
{
    int warp = threadIdx.x / 32 + blockIdx.x * (blockDim.x / 32);
    int lane = threadIdx.x % 32;
    int stride = gridDim.x * (blockDim.x / 32);

    for (int v = warp; v < nodes; v += stride)
    {
        float local_gradient = 0;
        for (int edge = offsets[v]+lane; edge < offsets[v+1]; edge += 32)
        {
            int u = edges[edge];
            local_gradient -= weights[edge]*(1-2*probabilities[u]);
        }

        for (int delta = 16; delta > 0; delta >>= 1)
        {
            local_gradient += __shfl_down_sync(0xffffffff, local_gradient, delta);
        }
        if (lane == 0)
        {
            node_gradients[v] = local_gradient;
        }
    }
}

__global__ void GnnKernels::score_backprop(float* probabilities, float* node_gradients, int nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < nodes; i += stride)
    {
        float prob = probabilities[i];
        float grad = node_gradients[i];

        node_gradients[i] = grad*prob*(1-prob);
    }
}

__global__ void GnnKernels::proj_backprop(float* output_layer, float* node_gradients, float* proj_gradients, int nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    __shared__ float l_proj_grads[FEATURE_DIM];

    for (int i = threadIdx.x; i < FEATURE_DIM; i += blockDim.x)
    {
        l_proj_grads[i] = 0;
    }
    __syncthreads();

    for (int i = idx; i < nodes*FEATURE_DIM; i += stride)
    {
        int grad_lane = i % FEATURE_DIM;
        float grad = node_gradients[i/FEATURE_DIM];
        float feature = output_layer[i];

        atomicAdd(&l_proj_grads[grad_lane], feature*grad);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < FEATURE_DIM; i += blockDim.x)
    {
        atomicAdd(&proj_gradients[i], l_proj_grads[i]);
    }
}

__global__ void GnnKernels::feature_backprop(float* feature_gradients, float* node_gradients, float* projection, int nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    __shared__ float l_proj[FEATURE_DIM];

    for (int i = threadIdx.x; i < FEATURE_DIM; i += blockDim.x)
    {
        l_proj[i] = projection[i];
    }
    __syncthreads();

    for (int i = idx; i < nodes*FEATURE_DIM; i += stride)
    {
        int grad_lane = i % FEATURE_DIM;
        float grad = node_gradients[i/FEATURE_DIM];

        feature_gradients[i] = grad * l_proj[grad_lane];
    }
}

__global__ void GnnKernels::tanh_backprop(float* feature_gradients, float* output_layer, float* projection, int nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < nodes*FEATURE_DIM; i += stride)
    {
        float grad = feature_gradients[i];
        float feature = output_layer[i];

        feature_gradients[i] = (1-feature*feature) * grad;
    }
}

__global__ void GnnKernels::weight_backprop(int* edges, int* offsets, float* features, float* feature_gradients, float* layer_weight_gradients, int nodes)
{
    int warp = threadIdx.x / 32 + blockIdx.x * (blockDim.x / 32);
    int lane = threadIdx.x % 32;
    int stride = gridDim.x * (blockDim.x / 32);

    for (int v = warp; v < nodes; v += stride)
    {
        for (int edge = offsets[v]+lane; edge < offsets[v+1]; edge += 32)
        {
            int u = edges[edge];
            float accumulator[FEATURE_DIM];
            for (int i = 0; i < FEATURE_DIM; i++)
            {
                for (int j = 0; j < FEATURE_DIM; j++)
                {
                    accumulator[j] = 0;
                }

                for (int j = 0; j < FEATURE_DIM; j++)
                {
                    accumulator[j] += features[u * FEATURE_DIM + i] * feature_gradients[v * FEATURE_DIM + j];
                }

                for (int delta = 16; delta > 0; delta >>= 1)
                {
                    for (int k = 0; k < FEATURE_DIM; k++)
                    {
                        unsigned mask = __activemask();
                        accumulator[k] += __shfl_down_sync(mask, accumulator[k], delta);
                    }
                }

                if (lane == 0)
                {
                    for (int k = 0; k < FEATURE_DIM; k++)
                    {
                        atomicAdd(&layer_weight_gradients[i*FEATURE_DIM + k], accumulator[k]);
                    }
                }

            }
        }
    }
}

__global__ void GnnKernels::update_weights(float* layer_weight_gradients, float* layer_weights, float* projection_gradients, 
    float* projection, float lr, int nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < FEATURE_DIM*FEATURE_DIM; i += stride)
    {
        layer_weights[i] -= layer_weight_gradients[i] * lr;
    }

    for (int i = idx; i < FEATURE_DIM; i += stride)
    {
        projection[i] -= projection_gradients[i] * lr;
    }
}

__global__ void GnnKernels::request_probabilities(float* probabilities, int offset, int nodes, int* requests, int* request_locations, 
    float* buffer, int buffer_offset, int request_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < request_size; i += stride)
    {
        int request = requests[i] - offset;
        if (request >= 0 && request < nodes)
        {
            buffer[request_locations[i] - buffer_offset] = probabilities[request];
        }
    }
}

__global__ void GnnKernels::fill_nan(float* buffer, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < size; i += stride)
    {
        buffer[i] = NAN;
    }
}

__global__ void GnnKernels::update_probabilities(float* probabilities, float* buffer, int nodes, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < nodes; i += stride)
    {
        if (i < offset)
        {
            continue;
        }
        float buff = buffer[i - offset];
        if (!isnan(buff))
        {
            probabilities[i] = buff;
        }
    }
}