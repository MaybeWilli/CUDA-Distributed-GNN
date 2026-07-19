#include "gpu_gnn.h"

#include "gnn.h"
#include "max_cut.h"
#include "gnn_kernels.cuh"

namespace
{
    float random_float(float a, float b)
    {
        static std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(a, b);
        return dist(rng);
    }
}

GpuGnn::GpuGnn(Graph& graph, int FEATURE_DIM, double lr, dim3 grid, dim3 block) : nodes(graph.nodes), FEATURE_DIM(FEATURE_DIM), lr(lr),
    grid(grid), block(block)
{
    vector<float> features(nodes*FEATURE_DIM);
    vector<float> layer_weights(FEATURE_DIM*FEATURE_DIM);
    vector<float> projection(FEATURE_DIM);

    for (int i = 0; i < nodes; i++)
    {
        int total_edges = 0;
        for (int edge = graph.offsets[i]; edge < graph.offsets[i+1]; edge++)
        {
            total_edges += graph.weights[edge];
        }
        features[i*FEATURE_DIM] = total_edges;
        features[i*FEATURE_DIM + 1] = graph.offsets[i+1]-graph.offsets[i];
        for (int j = 2; j < FEATURE_DIM; j++)
        {
            features[i*FEATURE_DIM + j] = random_float(-1, 1);
        }
    }

    for (int i = 0; i < FEATURE_DIM; i++)
    {
        projection[i] = random_float(-1, 1);
    }

    for (int i = 0; i < FEATURE_DIM*FEATURE_DIM; i++)
    {
        layer_weights[i] = random_float(-1, 1);
    }

    device = 0;
    cudaSetDevice(device);

    checkCuda ( cudaMalloc((void**)&d_edges, graph.offsets[nodes] * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_offsets, (nodes+1) * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_weights, graph.offsets[nodes] * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_features, nodes*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_output_layer, nodes*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_probabilities, nodes * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_layer_weights, FEATURE_DIM*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_projection, FEATURE_DIM * sizeof(float)));

    checkCuda ( cudaMalloc((void**)&d_node_gradients, nodes * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_feature_gradients, nodes*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_proj_gradients, FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_layer_weight_gradients, FEATURE_DIM*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&weight_buffer, FEATURE_DIM*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&proj_buffer, FEATURE_DIM * sizeof(float)));
    

    checkCuda( cudaMemcpy(d_edges, graph.edges.data(), graph.offsets[nodes] * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_offsets, graph.offsets.data(), (nodes+1) * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_weights, graph.weights.data(), graph.offsets[nodes] * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_features, features.data(), nodes*FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_layer_weights, layer_weights.data(), FEATURE_DIM*FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_projection, projection.data(), FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice) );

}

GpuGnn::GpuGnn(GraphPartition& partition, int partition_index, int FEATURE_DIM, double lr, vector<float>& layer_weights, vector<float>& projection, 
    vector<float>& global_features, int max_buff_size, dim3 grid, dim3 block) 
    : nodes(partition.active_node_count[partition_index]), FEATURE_DIM(FEATURE_DIM), lr(lr), grid(grid), block(block)
{
    Graph& graph = partition.graphs[partition_index];

    vector<float> features(graph.nodes*FEATURE_DIM);
    for (int i = 0; i < graph.nodes; i++)
    {
        int global_idx = i + partition.offsets[partition_index];
        if (i >= nodes)
        {
            global_idx = partition.local_to_global[partition_index][i];
        }
        for (int j = 0; j < FEATURE_DIM; j++)
        {
            features[i*FEATURE_DIM + j] = global_features[global_idx*FEATURE_DIM + j];
        }
    }

    device = 0;
    cudaSetDevice(device);

    checkCuda ( cudaMalloc((void**)&d_edges, graph.offsets[nodes] * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_offsets, (nodes+1) * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_weights, graph.offsets[nodes] * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_features, graph.nodes*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_output_layer, nodes*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_probabilities, graph.nodes * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_layer_weights, FEATURE_DIM*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_projection, FEATURE_DIM * sizeof(float)));

    checkCuda ( cudaMalloc((void**)&d_node_gradients, nodes * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_feature_gradients, nodes*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_proj_gradients, FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_layer_weight_gradients, FEATURE_DIM*FEATURE_DIM * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_request_buffer, max_buff_size * sizeof(float)));

    buff_size = graph.nodes - nodes;
    checkCuda ( cudaMalloc((void**)&d_local_buffer, buff_size * sizeof(float)));
    checkCuda ( cudaMalloc((void**)&d_buffer, buff_size * sizeof(float)));
    checkCuda( cudaMemset(d_buffer, 0, buff_size * sizeof(float)) );
    checkCuda( cudaMemset(d_local_buffer, 0, buff_size * sizeof(float)) );
    

    checkCuda( cudaMemcpy(d_edges, graph.edges.data(), graph.offsets[nodes] * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_offsets, graph.offsets.data(), (nodes+1) * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_weights, graph.weights.data(), graph.offsets[nodes] * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_features, features.data(), 
        features.size() * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_layer_weights, layer_weights.data(), FEATURE_DIM*FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_projection, projection.data(), FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice) );
}//*/

void GpuGnn::forward_run(bool should_print)
{
    checkCuda( cudaMemset(d_output_layer, 0, nodes*FEATURE_DIM * sizeof(float)) );
    checkCuda( cudaMemset(d_probabilities, 0, nodes * sizeof(float)) );
    GnnKernels::fill_nan<<<grid, block>>>(d_local_buffer, buff_size);
    checkCuda( cudaMemset(d_buffer, 0, buff_size * sizeof(float)) );
    GnnKernels::forward_run<<<grid, block>>>(d_features, d_edges, d_offsets, d_output_layer, d_probabilities, d_layer_weights, d_projection, nodes);
}

void GpuGnn::prob_backprop()
{
    checkCuda( cudaMemset(d_node_gradients, 0, nodes * sizeof(float)) );
    checkCuda( cudaMemset(d_feature_gradients, 0, nodes*FEATURE_DIM * sizeof(float)) );
    checkCuda( cudaMemset(d_proj_gradients, 0, FEATURE_DIM * sizeof(float)) );
    checkCuda( cudaMemset(d_layer_weight_gradients, 0, FEATURE_DIM*FEATURE_DIM * sizeof(float)) );


    GnnKernels::prob_backprop<<<grid, block>>>(d_edges, d_offsets, d_output_layer, d_probabilities, d_weights, d_layer_weights,
            d_projection, d_node_gradients, nodes);
    checkCuda(cudaGetLastError());
}

void GpuGnn::sigmoid_backprop()
{
    GnnKernels::score_backprop<<<grid, block>>>(d_probabilities, d_node_gradients, nodes);
}

void GpuGnn::proj_backprop()
{
    GnnKernels::proj_backprop<<<grid, block>>>(d_output_layer, d_node_gradients, d_proj_gradients, nodes);
    GnnKernels::feature_backprop<<<grid, block>>>(d_feature_gradients, d_node_gradients, d_projection, nodes);
}

void GpuGnn::tanh_backprop()
{
    GnnKernels::tanh_backprop<<<grid, block>>>(d_feature_gradients, d_output_layer, d_projection, nodes);
}

void GpuGnn::message_backprop()
{
    GnnKernels::weight_backprop<<<grid, block>>>(d_edges, d_offsets, d_features, d_feature_gradients, d_layer_weight_gradients, nodes);
}

void GpuGnn::update_weights()
{
    GnnKernels::update_weights<<<grid, block>>>(d_layer_weight_gradients, d_layer_weights, d_proj_gradients, d_projection, lr, nodes);
}

void GpuGnn::update_weights(vector<float>& projection_gradients, vector<float>& weight_gradients)
{
    cudaSetDevice(device);
    checkCuda( cudaMemcpy(d_proj_gradients, projection_gradients.data(), FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_layer_weight_gradients, weight_gradients.data(), FEATURE_DIM * FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice) );
    GnnKernels::update_weights<<<grid, block>>>(d_layer_weight_gradients, d_layer_weights, d_proj_gradients, d_projection, lr, nodes);
}

void GpuGnn::back_prop()
{
    prob_backprop();
    sigmoid_backprop();
    proj_backprop();
    tanh_backprop();
    message_backprop();
    update_weights();
}

void GpuGnn::request_probabilities(int* requests, int* request_locations, int offset, float* buffer, int request_device, 
            int request_size, int buffer_offset, ncclComm_t comm, bool should_print)
{
    cudaSetDevice(device);
    //using nccl for demo purposes
    GnnKernels::fill_nan<<<grid, block>>>(d_request_buffer, request_size);
    //checkCuda ( cudaDeviceSynchronize() );
    GnnKernels::request_probabilities<<<grid, block>>>(d_probabilities, offset, nodes, requests, request_locations, d_request_buffer, 
        buffer_offset, request_size);
    checkCuda ( cudaDeviceSynchronize() );
    vector<float> test(request_size);

    ncclReduce(d_request_buffer, buffer, request_size, ncclFloat, ncclSum, request_device, comm, 0);
    checkCuda ( cudaDeviceSynchronize() );
}

void GpuGnn::save_to_buffer()
{
    cudaSetDevice(device);
    GnnKernels::update_probabilities<<<grid, block>>>(d_local_buffer, d_buffer, buff_size, 0);
}

void GpuGnn::clear_buffer()
{
    cudaSetDevice(device);
    checkCuda( cudaMemset(d_buffer, 0, buff_size * sizeof(float)) );
}

void GpuGnn::flush_buffer()
{
    cudaSetDevice(device);
    GnnKernels::update_probabilities<<<grid, block>>>(d_probabilities, d_local_buffer, nodes+buff_size, nodes);
}