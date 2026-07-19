nvcc -O3  src/distributed_gpu_gnn.cu \
    src/graph.cpp \
    src/gnn_kernels.cu \
    src/gpu_gnn.cu \
    src/gnn.cpp \
    src/graph_utils.cpp \
    src/max_cut.cpp \
    src/training_utils.cu -lnccl -o gnn

g++ -O3 -march=native src/baseline_gnn.cpp \
    src/graph.cpp \
    src/gnn.cpp \
    src/graph_utils.cpp \
    src/max_cut.cpp -o cpu_gnn
