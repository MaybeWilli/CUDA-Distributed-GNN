#include <stdio.h>
#include <assert.h>
#include <array>
#include <iostream>
#include <vector>
#include "graph.h"
#include <chrono>
#include "gnn_kernels.cuh"

#include "gpu_gnn.h"
#include "gnn.h"
#include <nccl.h>
#include "max_cut.h"
#include "training_utils.cuh"

int main(int argc, char** argv)
{
    srand(12345);

    int iterations = 100;
    int nodes = 30000;
    int partitions = 4;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--nodes" && i + 1 < argc)
        {
            nodes = std::stoi(argv[++i]);
        }
        else if (arg == "--partitions" && i + 1 < argc)
        {
            partitions = std::stoi(argv[++i]);
        }
        else if (arg == "--iterations" && i + 1 < argc)
        {
            iterations = std::stoi(argv[++i]);
        }
        else if (arg == "--help")
        {
            std::cout
                << "Usage:\n"
                << "  ./gnn --nodes N --partitions P --iterations I\n\n"
                << "Options:\n"
                << "  --nodes        Number of graph vertices\n"
                << "  --partitions   Number of graph partitions\n"
                << "  --iterations   Training iterations\n";
            exit(0);
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            exit(1);
        }
    }
    Graph graph;
    create_graph(graph, nodes, 2);
    //int FEATURE_DIM = 8;
    double lr = 0.01;
    int threads = 256;
    int num_devices = 1;
    dim3 block(threads, 1);
    dim3 grid(nodes/threads + 1, 1);
    Gnn gnn = Gnn(graph, FEATURE_DIM, lr);
    vector<float> total_probabilities(nodes, 0);

    //partition the graph
    GraphPartition partition;
    create_partition(partition, graph, partitions);

    int biggest_buffer = 0;
    for (int i = 0; i < partitions; i++)
    {
        if (partition.graphs[i].nodes - partition.active_node_count[i] > biggest_buffer)
        {
            biggest_buffer = partition.graphs[i].nodes - partition.active_node_count[i];
        }
    }
    vector<GpuGnn> gnns;
    for (int i = 0; i < partitions; i++)
    {
        gnns.emplace_back(partition, i, FEATURE_DIM, lr, gnn.weights, gnn.projection, gnn.features, biggest_buffer, grid, block);
    }

    float milliseconds = training(partition, partitions, gnns, total_probabilities, num_devices, iterations, biggest_buffer);

    cout<<"Vertices: "<<nodes<<endl;
    cout<<"Iterations: "<<iterations<<endl;
    cout<<"Partitions: "<<partitions<<endl;
    cout << "Run time: " << milliseconds << " ms"<<endl;
    cout<<"Time per iteration: "<<milliseconds/iterations<<" ms"<<endl;
    
    int config_size = int(nodes/32) + 1;
    vector<int> config(config_size);
    int total_edges = 0;
    for (int i = 0; i < graph.offsets[nodes]; i++)
    {
        total_edges += graph.weights[i];
    }

    for (int i = 0; i < nodes; i++)
    {
        if (total_probabilities[i] >= 0.5f)
        {
            config[int(i/32)] |= (1u << (i % 32));
        }
    }

    MaxCut max_cut(graph, config);
    MaxCut max_cut2(graph);
    cout<<"GNN max cut weight: "<<max_cut.get_weight()<<endl;;
    cout<<"Random initial weight: "<<max_cut2.get_weight()<<endl;;
    cout<<"Total edges: "<<total_edges<<endl;//*/
}