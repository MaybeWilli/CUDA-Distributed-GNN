#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include "graph.h"
#include <unordered_map>

struct NodeMap
{
    int globalId;
    int localId;
};

struct GraphPartition
{
    vector<Graph> graphs;
    vector<int> active_nodes;
    vector<int> active_node_count;
    vector<unordered_map<int, int>> global_to_local;
    vector<unordered_map<int, int>> local_to_global;
    vector<vector<int>> request_ids;
    vector<vector<int>> request_locations;
    vector<int> offsets;
    Graph graph;
    
};

void create_partition(GraphPartition& partition, Graph& graph, int partitions);

#endif