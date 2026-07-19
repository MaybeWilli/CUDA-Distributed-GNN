#include "gnn.h"
#include <chrono>

int main()
{
    srand(12345);
    Graph graph;
    int nodes = 30000;
    create_graph(graph, nodes, 2);
    int FEATURE_DIM = 16;
    double lr = 0.01;

    Gnn gnn = Gnn(graph, FEATURE_DIM, lr);
    int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++)
    {
        gnn.forward_run();
        gnn.back_prop();
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    );
    float milliseconds = elapsed.count();
    cout<<"Vertices: "<<nodes<<endl;
    cout<<"Iterations: "<<iterations<<endl;
    cout << "Run time: " << milliseconds << " ms"<<endl;
    cout<<"Time per iteration: "<<milliseconds/iterations<<" ms"<<endl;
    gnn.get_results();
}