#ifndef MAX_CUT_H
#define MAX_CUT_H

#include <vector>
#include <random>
#include <iostream>
#include "graph.h"

using namespace std;

class MaxCut
{
    public:
        int weight;
        Graph graph;
        int config_size = 0;
        vector<int> config;
        vector<int> gains;

        vector<int> max_config;
        int max_weight;

        MaxCut(Graph& graph);
        MaxCut(Graph& graph, vector<int>& config);
        bool solve();
        int calculate_gain(int candidate);
        void flip_vertex(int index);
        int get_weight();

        void save_max_config();
        int get_max_weight();

        int perfect_solve();
};

#endif