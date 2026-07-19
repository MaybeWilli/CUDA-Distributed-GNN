#ifndef GPU_GNN_H
#define GPU_GNN_H

#include <vector>
#include "graph.h"
#include "graph_utils.h"
#include <unordered_map>
#include <cuda_runtime.h>
#include <nccl.h>

using namespace std;

class GpuGnn
{
    public:
        //Graph graph;
        int FEATURE_DIM;
        int nodes;
        double lr;
        /*vector<float> features;
        vector<float> weights;
        vector<float> projection;
        vector<float> output_layer;
        vector<float> probabilities;

        vector<float> node_gradients;
        vector<float> feature_gradients;
        vector<float> weight_gradients;
        vector<float> projection_gradients;*/

        int* d_edges;
        int* d_offsets;
        int* d_weights;
        float* d_features;
        float* d_output_layer;
        float* d_probabilities;
        float* d_layer_weights;
        float* d_projection;

        float* d_node_gradients;
        float* d_feature_gradients;
        float* d_proj_gradients;
        float* d_layer_weight_gradients;

        float* d_local_buffer;
        float* d_buffer;
        int buff_size = 0;

        float* proj_buffer;
        float* weight_buffer;

        float* d_request_buffer;

        dim3 grid;
        dim3 block;
        int device;

        GpuGnn(Graph& graph, int FEATURE_DIM, double lr, dim3 grid, dim3 block);
        GpuGnn(GraphPartition& partition, int partition_index, int FEATURE_DIM, double lr, 
            vector<float>& weights, vector<float>& projection, vector<float>& global_features, 
            int max_buff_size, dim3 grid, dim3 block);

        void forward_run(bool should_print=false);
        void prob_backprop();
        void sigmoid_backprop();
        void proj_backprop();
        void tanh_backprop();
        void message_backprop();
        void update_weights();
        void update_weights(vector<float>& projection_gradients, vector<float>& weight_gradients);
        void back_prop();

        void save_to_buffer();
        void clear_buffer();
        void flush_buffer();

        void request_probabilities(int* requests, int* request_locations, int offset, float* buffer, int request_device, 
            int request_size, int buffer_offset, ncclComm_t comm, bool should_print=false);
};

#endif