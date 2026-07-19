#include "gnn.h"
#include "max_cut.h"

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

Gnn::Gnn(Graph& graph, int FEATURE_DIM, double lr) : graph(graph), FEATURE_DIM(FEATURE_DIM), lr(lr), nodes(graph.nodes),
    features(graph.nodes * FEATURE_DIM), weights(FEATURE_DIM*FEATURE_DIM), projection(FEATURE_DIM), output_layer(graph.nodes * FEATURE_DIM),
    probabilities(nodes), node_gradients(nodes), feature_gradients(nodes * FEATURE_DIM), weight_gradients(FEATURE_DIM * FEATURE_DIM),
    projection_gradients(FEATURE_DIM)
{
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
        for (int j = 0; j < FEATURE_DIM; j++)
        {
            weights[i*FEATURE_DIM + j] = random_float(-1, 1);
        }
    }

}

Gnn::Gnn(GraphPartition& partition, int partition_index, int FEATURE_DIM, double lr, vector<float>& weights, vector<float>& projection, 
    vector<float>& global_features)
    : graph(partition.graphs[partition_index]), FEATURE_DIM(FEATURE_DIM), lr(lr), nodes(partition.active_node_count[partition_index]),
    features(graph.nodes * FEATURE_DIM), weights(weights), projection(projection), output_layer(graph.nodes * FEATURE_DIM),
    probabilities(graph.nodes), node_gradients(nodes), feature_gradients(nodes * FEATURE_DIM), weight_gradients(FEATURE_DIM * FEATURE_DIM),
    projection_gradients(FEATURE_DIM)
{
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
}

void Gnn::forward_run(bool should_print)
{
    for (int i = 0; i < nodes*FEATURE_DIM; i++)
    {
        output_layer[i] = 0;
    }

    for (int v = 0; v < nodes; v++)
    {
        for (int edge = graph.offsets[v]; edge < graph.offsets[v+1]; edge++)
        {
            int u = graph.edges[edge];

            for (int j = 0; j < FEATURE_DIM; j++)
            {
                for (int k = 0; k < FEATURE_DIM; k++)
                {
                    output_layer[v*FEATURE_DIM + j] += features[u*FEATURE_DIM + k] * weights[j*FEATURE_DIM + k];
                }
            }

        }

        int deg = graph.offsets[v+1] - graph.offsets[v];
        for (int i = 0; i < FEATURE_DIM; i++)
        {
            output_layer[v*FEATURE_DIM + i] = tanh(output_layer[v*FEATURE_DIM + i]);
        }
    }

    //projection layer
    for (int i = 0; i < nodes; i++)
    {
        float score = 0;
        for (int j = 0; j < FEATURE_DIM; j++)
        {
            score += output_layer[i*FEATURE_DIM + j] * projection[j];
        }
        probabilities[i] = sigmoid(score);
        if (should_print)
        {
            cout<<probabilities[i]<<" "<<score<<endl;//<<" "<<output_layer[i].feature1<<" "<<output_layer[i].degree<<endl;
        }
    }
}

void Gnn::prob_backprop()
{
    std::fill(node_gradients.begin(), node_gradients.end(), 0);
    
    for (int v = 0; v < nodes; v++)
    {
        for (int edge = graph.offsets[v]; edge < graph.offsets[v+1]; edge++)
        {
            int u = graph.edges[edge];

            node_gradients[v] -= graph.weights[edge]*(1-2*probabilities[u]);
        }
    }
}

void Gnn::sigmoid_backprop()
{
    for (int v = 0; v < nodes; v++)
    {
        node_gradients[v] = node_gradients[v]*probabilities[v]*(1-probabilities[v]);
    }
}

void Gnn::proj_backprop()
{
    std::fill(projection_gradients.begin(), projection_gradients.end(), 0);
    for (int i = 0; i < nodes; i++)
    {
        for (int j = 0; j < FEATURE_DIM; j++)
        {
            projection_gradients[j] += output_layer[i*FEATURE_DIM + j] * node_gradients[i];
        }
    }

    std::fill(feature_gradients.begin(), feature_gradients.end(), 0);
    for (int i = 0; i < nodes; i++)
    {
        for (int j = 0; j < FEATURE_DIM; j++)
        {
            feature_gradients[i*FEATURE_DIM + j] = projection[j] * node_gradients[i];
        }
    }
}

void Gnn::tanh_backprop()
{
    for (int i = 0; i < nodes; i++)
    {
        int deg = graph.offsets[i+1] - graph.offsets[i];
        for (int j = 0; j < FEATURE_DIM; j++)
        {
            feature_gradients[i*FEATURE_DIM + j] = (1-output_layer[i*FEATURE_DIM + j]*output_layer[i*FEATURE_DIM + j])
                 * feature_gradients[i*FEATURE_DIM + j];// / deg;
        }
    }
}

void Gnn::message_backprop()
{
    std::fill(weight_gradients.begin(), weight_gradients.end(), 0);
    for (int v = 0; v < nodes; v++)
    {
        for (int edge = graph.offsets[v]; edge < graph.offsets[v+1]; edge++)
        {
            int u = graph.edges[edge];

            for (int i = 0; i < FEATURE_DIM; i++)
            {
                for (int j = 0; j < FEATURE_DIM; j++)
                {
                    weight_gradients[i*FEATURE_DIM + j] += features[u*FEATURE_DIM + i] * feature_gradients[v*FEATURE_DIM + j];
                }
            }
        }
    }
}

void Gnn::update_weights()
{
    for (int i = 0; i < FEATURE_DIM; i++)
    {
        projection[i] -= projection_gradients[i] * lr;
    }

    for (int i = 0; i < FEATURE_DIM * FEATURE_DIM; i++)
    {
        weights[i] -= weight_gradients[i] * lr;
    }
}

void Gnn::update_weights(vector<float>& projection_gradients, vector<float>& weight_gradients)
{
    for (int i = 0; i < FEATURE_DIM; i++)
    {
        projection[i] -= projection_gradients[i] * lr;
    }

    for (int i = 0; i < FEATURE_DIM * FEATURE_DIM; i++)
    {
        weights[i] -= weight_gradients[i] * lr;
    }
}

void Gnn::back_prop()
{
    prob_backprop();
    sigmoid_backprop();
    proj_backprop();
    tanh_backprop();
    message_backprop();
    update_weights();
}

void Gnn::get_results()
{
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
    cout<<"Total edges: "<<total_edges<<endl;//*/
}

void Gnn::request_output(vector<int>& requests, vector<int>& request_locations, int offset, vector<float>& buffer)
{
    for (int i = 0; i < requests.size(); i++)
    {
        int request = requests[i] - offset;
        if (request >= 0 && request < nodes)
        {
            for (int j = 0; j < FEATURE_DIM; j++)
            {
                buffer[request_locations[i]*FEATURE_DIM + j] = output_layer[request*FEATURE_DIM + j];
            }
        }
    }
}

void Gnn::request_features(vector<int>& requests, vector<int>& request_locations, int offset, vector<float>& buffer)
{
    for (int i = 0; i < requests.size(); i++)
    {
        int request = requests[i] - offset;
        if (request >= 0 && request < nodes)
        {
            for (int j = 0; j < FEATURE_DIM; j++)
            {
                buffer[request_locations[i]*FEATURE_DIM + j] = feature_gradients[request*FEATURE_DIM + j];
            }
        }
    }
}

void Gnn::request_probabilities(vector<int>& requests, vector<int>& request_locations, int offset, vector<float>& buffer, bool should_print)
{
    for (int i = 0; i < requests.size(); i++)
    {
        int request = requests[i] - offset;
        if (should_print)
        {
            cout<<requests[i]<<" "<<offset<<" "<<request<<endl;
        }
        if (request >= 0 && request < nodes)
        {
            buffer[request_locations[i]] = probabilities[request];
        }
    }
}