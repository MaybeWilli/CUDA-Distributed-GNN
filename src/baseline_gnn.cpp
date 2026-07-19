#include "gnn.h"
#include <chrono>

int main(int argc, char** argv)
{
    srand(12345);
    Graph graph;
    int iterations = 100;
    int nodes = 30000;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--nodes" && i + 1 < argc)
        {
            nodes = std::stoi(argv[++i]);
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
    create_graph(graph, nodes, 2);
    int FEATURE_DIM = 16;
    double lr = 0.01;

    Gnn gnn = Gnn(graph, FEATURE_DIM, lr);
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