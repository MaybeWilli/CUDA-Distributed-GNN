#include "graph.h"

void create_undirected_graph(vector<int>& temp_offsets, vector<int>& temp_edges, vector<int>& temp_weights, Graph& graph)
{
    vector<int> deg(graph.nodes+1);
    for (int v = 0; v < graph.nodes; v++)
    {
        for (int j = temp_offsets[v]; j < temp_offsets[v+1]; j++)
        {
            int u = temp_edges[j];
            deg[u]++;
            deg[v]++;
        }
    }

    int total = 0;
    graph.offsets.resize(graph.nodes+1);
    vector<int> indexes(graph.nodes+1);
    for (int i = 0; i < graph.nodes; i++)
    {
        graph.offsets[i] = total;
        total += deg[i];
        indexes[i] = graph.offsets[i];
    }

    int total_edges = temp_offsets[graph.nodes]*2;
    graph.offsets[graph.nodes] = total_edges;
    graph.edges.resize(total_edges);
    graph.weights.resize(total_edges);
    

    for (int v = 0; v < graph.nodes; v++)
    {
        for (int j = temp_offsets[v]; j < temp_offsets[v+1]; j++)
        {
            int u = temp_edges[j];
            int weight = temp_weights[j];
            graph.edges[indexes[u]] = v;
            graph.edges[indexes[v]] = u;
            graph.weights[indexes[u]] = weight;
            graph.weights[indexes[v]] = weight;
            indexes[u]++;
            indexes[v]++;
        }
    }
}

void create_graph(Graph& graph, int nodes, int mode)
{
    graph.nodes = nodes;
    vector<int> temp_edges;
    vector<int> temp_offsets;
    vector<int> temp_weights;
    if (mode == 0)
    {
        vector<int> deg(nodes);
        int edge_total = 0;
        for (int i = 0; i < nodes; i++)
        {
            //deg[i] = rand() % 4 + 4;
            deg[i] = rand() % 3 + 1;
            edge_total += deg[i];
        }

        temp_edges.resize(edge_total);
        temp_weights.resize(edge_total);
        temp_offsets.resize(nodes+1);
        int offset = 0;
        for (int i = 0; i < nodes; i++)
        {
            temp_offsets[i] = offset;
            for (int j = 0; j < deg[i]; j++)
            {
                int v = rand() % nodes;
                if (v == i)
                {
                    v = (v + 2) % nodes;
                }
                temp_edges[offset + j] = v;
                temp_weights[offset + j] = rand() % 10 + 1;
            }
            offset += deg[i];

        }
        temp_offsets[nodes] = edge_total;

    }
    else if (mode == 1)
    {
        int l = std::sqrt(nodes);
        temp_edges.resize(nodes*2);
        temp_offsets.resize(nodes+1);
        temp_weights.resize(nodes*2);

        for (int i = 0; i < nodes*2; i += 2)
        {
            temp_offsets[i/2] = i;
            temp_edges[i] = (i/4 % l + 1) % l + (int(i/4/l)) * l;
            temp_edges[i+1] = (i/4 % l) + ((int(i/4/l)+1) % l) * l;
            temp_weights[i] = rand() % 10 + 1;
            temp_weights[i+1] = rand() % 10 + 1;
        }

        temp_offsets[nodes] = nodes*2;
    }
    else if (mode == 2)
    {
        vector<int> deg(nodes);

        int stub_count = 0;
        double gamma = 2.5;

        for (int i = 0; i < nodes; i++)
        {
            double u = (rand() + 1.0) / (RAND_MAX + 1.0);

            deg[i] = std::max(
                1,
                (int)pow(u, -1.0 / (gamma - 1))
            );

            deg[i] = std::min(
                deg[i],
                int(sqrt(nodes))
            );

            stub_count += deg[i];
        }

        if (stub_count & 1)
        {
            deg[rand() % nodes]++;
            stub_count++;
        }

        std::vector<int> stubs;
        stubs.reserve(stub_count);

        for (int v = 0; v < nodes; v++)
        {
            for (int j = 0; j < deg[v]; j++)
            {
                stubs.push_back(v);
            }
        }

        std::mt19937 rng(rand());
        std::shuffle(stubs.begin(), stubs.end(), rng);

        int edge_total = stub_count / 2;

        temp_edges.resize(edge_total);
        temp_weights.resize(edge_total);
        temp_offsets.resize(nodes + 1);

        for (int i = 0; i < stub_count; i += 2)
        {
            int u = stubs[i];
            temp_offsets[u + 1]++;
        }

        for (int i = 1; i <= nodes; i++)
        {
            temp_offsets[i] += temp_offsets[i - 1];
        }

        std::vector<int> ptr(nodes);

        for (int i = 0; i < nodes; i++)
        {
            ptr[i] = temp_offsets[i];
        }

        for (int i = 0; i < stub_count; i += 2)
        {
            int u = stubs[i];
            int v = stubs[i + 1];

            if (u == v)
            {
                v = (v + 1) % nodes;
            }

            int pos = ptr[u]++;

            temp_edges[pos] = v;
            temp_weights[pos] = rand() % 10 + 1;
        }

    }//*/
    create_undirected_graph(temp_offsets, temp_edges, temp_weights, graph);
}