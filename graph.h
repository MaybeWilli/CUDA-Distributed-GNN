#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

struct Graph
{
    int nodes;
    vector<int> edges;
    vector<int> offsets;
    vector<int> weights;
};

void create_graph(Graph& graph, int nodes, int mode);

#endif