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

    float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
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
    //vector<float> layer_weights(FEATURE_DIM*FEATURE_DIM);
    //vector<float> projection(FEATURE_DIM);
    Graph& graph = partition.graphs[partition_index];

    /*for (int i = 0; i < nodes; i++)
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
    }*/

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
    //checkCuda ( cudaDeviceSynchronize() );
    checkCuda( cudaMemset(d_buffer, 0, buff_size * sizeof(float)) );
    //checkCuda( cudaMemset(d_local_buffer, 0, buff_size * sizeof(float)) );
    //checkCuda(cudaDeviceSynchronize());
    GnnKernels::forward_run<<<grid, block>>>(d_features, d_edges, d_offsets, d_output_layer, d_probabilities, d_layer_weights, d_projection, nodes);
    //checkCuda(cudaGetLastError());
    //checkCuda(cudaDeviceSynchronize());
}

void GpuGnn::prob_backprop()
{
    dim3 grid2(1, 1);
    dim3 block2(32, 1);
    checkCuda( cudaMemset(d_node_gradients, 0, nodes * sizeof(float)) );
    checkCuda( cudaMemset(d_feature_gradients, 0, nodes*FEATURE_DIM * sizeof(float)) );
    checkCuda( cudaMemset(d_proj_gradients, 0, FEATURE_DIM * sizeof(float)) );
    checkCuda( cudaMemset(d_layer_weight_gradients, 0, FEATURE_DIM*FEATURE_DIM * sizeof(float)) );
    //checkCuda(cudaDeviceSynchronize());

    /*cout<<"Probabilities:"<<endl;
    vector<float> probs(nodes+buff_size, 0);
    cudaMemcpy(probs.data(), d_probabilities, probs.size()*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < probs.size(); i++)
    {
        cout<<probs[i]<<endl;
    }
    cout<<"--------------"<<endl;*/


    GnnKernels::prob_backprop<<<grid, block>>>(d_edges, d_offsets, d_output_layer, d_probabilities, d_weights, d_layer_weights,
            d_projection, d_node_gradients, nodes);
    checkCuda(cudaGetLastError());
    //checkCuda(cudaDeviceSynchronize());
}

void GpuGnn::sigmoid_backprop()
{
    dim3 grid2(1, 1);
    dim3 block2(32, 1);
    GnnKernels::score_backprop<<<grid, block>>>(d_probabilities, d_node_gradients, nodes);
    //checkCuda(cudaGetLastError());
    //checkCuda(cudaDeviceSynchronize());
}

void GpuGnn::proj_backprop()
{
    dim3 grid2(1, 1);
    dim3 block2(32, 1);
    GnnKernels::proj_backprop<<<grid, block>>>(d_output_layer, d_node_gradients, d_proj_gradients, nodes);
    //checkCuda(cudaGetLastError());
    //checkCuda(cudaDeviceSynchronize());
    GnnKernels::feature_backprop<<<grid, block>>>(d_feature_gradients, d_node_gradients, d_projection, nodes);
    //checkCuda(cudaGetLastError());
    //checkCuda(cudaDeviceSynchronize());
}

void GpuGnn::tanh_backprop()
{
    dim3 grid2(1, 1);
    dim3 block2(32, 1);
    GnnKernels::tanh_backprop<<<grid, block>>>(d_feature_gradients, d_output_layer, d_projection, nodes);
    //checkCuda(cudaGetLastError());
    //checkCuda(cudaDeviceSynchronize());
}

void GpuGnn::message_backprop()
{
    dim3 grid2(1, 1);
    dim3 block2(32, 1);
    GnnKernels::weight_backprop<<<grid, block>>>(d_edges, d_offsets, d_features, d_feature_gradients, d_layer_weight_gradients, nodes);
    //checkCuda(cudaGetLastError());
    //checkCuda(cudaDeviceSynchronize());
}

void GpuGnn::update_weights()
{

    //checkCuda(cudaDeviceSynchronize());
    GnnKernels::update_weights<<<grid, block>>>(d_layer_weight_gradients, d_layer_weights, d_proj_gradients, d_projection, lr, nodes);
    //checkCuda(cudaGetLastError());
    //checkCuda(cudaDeviceSynchronize());
}

void GpuGnn::update_weights(vector<float>& projection_gradients, vector<float>& weight_gradients)
{
    cudaSetDevice(device);
    checkCuda( cudaMemcpy(d_proj_gradients, projection_gradients.data(), FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_layer_weight_gradients, weight_gradients.data(), FEATURE_DIM * FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice) );
    //checkCuda(cudaDeviceSynchronize());
    GnnKernels::update_weights<<<grid, block>>>(d_layer_weight_gradients, d_layer_weights, d_proj_gradients, d_projection, lr, nodes);
    checkCuda(cudaGetLastError());
    //checkCuda(cudaDeviceSynchronize());
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

void GpuGnn::get_results()
{
    /*vector<float> probabilities(nodes);
    int config_size = int(nodes/32) + 1;
    vector<int> config(config_size);
    int total_edges = 0;
    for (int i = 0; i < graph.offsets[nodes]; i++)
    {
        total_edges += graph.weights[i];
    }

    for (int i = 0; i < nodes; i++)
    {
        if (probabilities[i] >= random_float(0, 1.0f))
        {
            config[int(i/32)] |= (1u << (i % 32));
        }
    }

    MaxCut max_cut(graph, config);
    cout<<"Max cut weight: "<<max_cut.get_weight()<<endl;;
    cout<<"Total edges: "<<total_edges<<endl;*/
}

void GpuGnn::request_probabilities(int* requests, int* request_locations, int offset, float* buffer, int request_device, 
            int request_size, int buffer_offset, ncclComm_t comm, bool should_print)
{
    cudaSetDevice(device);
    //float* temp;
    //using nccl for demo purposes
    //checkCuda ( cudaMalloc((void**)&temp, request_size * sizeof(float)));
    GnnKernels::fill_nan<<<grid, block>>>(d_request_buffer, request_size);
    //checkCuda ( cudaDeviceSynchronize() );
    GnnKernels::request_probabilities<<<grid, block>>>(d_probabilities, offset, nodes, requests, request_locations, d_request_buffer, 
        buffer_offset, request_size);
    checkCuda ( cudaDeviceSynchronize() );
    vector<float> test(request_size);
    /*cudaMemcpy(test.data(), buffer, request_size*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < test.size(); i++)
    {
        cout<<"temp: "<<test[i]<<endl;
    }*/

    ncclReduce(d_request_buffer, buffer, request_size, ncclFloat, ncclSum, request_device, comm, 0);
    checkCuda ( cudaDeviceSynchronize() );

    /*cudaMemcpy(test.data(), buffer, request_size*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < test.size(); i++)
    {
        cout<<"d_buffer: "<<test[i]<<endl;
    }*/

    //checkCuda (cudaFree(temp));
    //checkCuda ( cudaDeviceSynchronize() );
}

void GpuGnn::save_to_buffer()
{
    cudaSetDevice(device);
    //cout<<"Saving to buffer"<<endl;
    GnnKernels::update_probabilities<<<grid, block>>>(d_local_buffer, d_buffer, buff_size, 0);
    //checkCuda ( cudaDeviceSynchronize() );
}

void GpuGnn::clear_buffer()
{
    cudaSetDevice(device);
    checkCuda( cudaMemset(d_buffer, 0, buff_size * sizeof(float)) );
    //checkCuda ( cudaDeviceSynchronize() );
}

void GpuGnn::flush_buffer()
{
    cudaSetDevice(device);
    //cout<<"Flushing buffer -------------------"<<endl;
    GnnKernels::update_probabilities<<<grid, block>>>(d_probabilities, d_local_buffer, nodes+buff_size, nodes);
    //checkCuda ( cudaDeviceSynchronize() );
}