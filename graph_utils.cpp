#include "graph_utils.h"
#include <set>

void create_partition(GraphPartition& partition, Graph& graph, int partitions)
{
    partition.graph = graph;
    for (int i = 0; i < partitions; i++)
    {
        partition.active_nodes.push_back(double(i+1)/partitions*graph.nodes);
    }
    

    int start = 0;
    int edge_start = 0;
    //partition.map.resize(partitions);
    for (int i = 0; i < partitions; i++)
    {
        partition.graphs.push_back(Graph());
        partition.offsets.push_back(start);
        Graph& curr_graph = partition.graphs[i];
        curr_graph.nodes = partition.active_nodes[i] - start;
        partition.active_node_count.push_back(curr_graph.nodes);
        curr_graph.offsets.reserve(partition.graphs[i].nodes+1);
        curr_graph.edges.reserve(graph.offsets[partition.active_nodes[i]] - edge_start);
        curr_graph.weights.reserve(graph.offsets[partition.active_nodes[i]] - edge_start);

        for (int j = start; j < partition.active_nodes[i]; j++)
        {
            partition.graphs[i].offsets.push_back(graph.offsets[j] - edge_start);
        }
        curr_graph.offsets.push_back(
            graph.offsets[partition.active_nodes[i]] - edge_start
        );

        for (int j = edge_start; j < graph.offsets[partition.active_nodes[i]]; j++)
        {
            partition.graphs[i].edges.push_back(graph.edges[j] - start);
            partition.graphs[i].weights.push_back(graph.weights[j]);
        }

        //cache nodes
        std::unordered_map<int, int> caches;
        std::unordered_map<int, int> r_caches;
        for (int j = 0; j < curr_graph.edges.size(); j++)
        {
            if (curr_graph.edges[j] < 0 || curr_graph.edges[j] >= curr_graph.nodes)
            {
                int idx = curr_graph.nodes + caches.size();
                auto [it, inserted] = caches.emplace(
                    curr_graph.edges[j] + start,
                    curr_graph.nodes + caches.size()
                );
                
                if (inserted)
                {
                    r_caches.emplace(it->second, it->first);
                }
            }
        }
        for (int j = 0; j < curr_graph.edges.size(); j++)
        {
            auto it = caches.find(curr_graph.edges[j] + start);
            if (it != caches.end())
            {
                curr_graph.edges[j] = it->second;
            }
        }
        partition.request_ids.push_back({});
        partition.request_locations.push_back({});
        for (int j = 0; j < r_caches.size(); j++)
        {
            auto it = r_caches.find(j + curr_graph.nodes);
            if (it != r_caches.end())
            {
                partition.request_ids[i].push_back(it->second);
                partition.request_locations[i].push_back(it->first);
            }
        }
        partition.global_to_local.push_back(caches);
        partition.local_to_global.push_back(r_caches);
        start = partition.active_nodes[i];
        edge_start = graph.offsets[partition.active_nodes[i]];
        curr_graph.nodes += caches.size();
    }
}