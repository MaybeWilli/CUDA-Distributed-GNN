#ifndef GNN_H
#define GNN_H

#include <vector>
#include "graph.h"
#include "graph_utils.h"
#include <unordered_map>

using namespace std;

class Gnn
{
    public:
        Graph graph;
        int FEATURE_DIM;
        int nodes;
        double lr;
        vector<float> features;
        vector<float> weights;
        vector<float> projection;
        vector<float> output_layer;
        vector<float> probabilities;

        vector<float> node_gradients;
        vector<float> feature_gradients;
        vector<float> weight_gradients;
        vector<float> projection_gradients;

        Gnn(Graph& graph, int FEATURE_DIM, double lr);
        Gnn(GraphPartition& partition, int partition_index, int FEATURE_DIM, double lr, 
            vector<float>& weights, vector<float>& projection, vector<float>& global_features);

        void forward_run(bool should_print=false);
        void prob_backprop();
        void sigmoid_backprop();
        void proj_backprop();
        void tanh_backprop();
        void message_backprop();
        void update_weights();
        void update_weights(vector<float>& projection_gradients, vector<float>& weight_gradients);
        void back_prop();
        void get_results();

        void request_output(vector<int>& requests, vector<int>& request_locations, int offset, vector<float>& buffer);
        void request_features(vector<int>& requests, vector<int>& request_locations, int offset, vector<float>& buffer);
        void request_probabilities(vector<int>& requests, vector<int>& request_locations, int offset, vector<float>& buffer, bool should_print=false);
};

#endif