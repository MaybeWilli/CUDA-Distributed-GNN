# CUDA-Distributed-GNN
Framework for distributed training for graph neural nets (GNNs) across multiple GPUs, written in CUDA/C++. Includes custom kernels for forward pass and backpropagation. GNNs are a type of neural net that use message passing so that every layer, each vertex gathers feature data from its neighbors, and adds it to its own. However, sometimes the original graph is too large to fit on a single GPU, and must be partitioned across multiple GPUs. This framework handles the partitioning and exchanging of information between different partitions. Currently, as a demo, it runs on a single GPU, but is meant to run on multiple.

## System Overview
The graph is split into n partitioned graphs, each in CSR format for faster GPU memory access. Each partitioned subgraph has format (active nodes | halos), where halos are read-only nodes that the partition's owned nodes have edges leading to, but do not own. These must be "refreshed" every iteration as partitions exchange halo nodes as they modify their data. Halo exchange is performed using NCCL reduceAll, though this framework only uses a single GPU for demo purposes, so reduceAll calls memcpyAsync in its backend to handle the single rank. Additionally, gradient aggregation between partitions at the end is done using memcpy, though with multiple ranks it would be faster to do with ncclReduceAll. NCCL treats different GPUs as different ranks, but because of hardware limitations, ncclReduceAll cannot be used in this manner efficiently.

Because all partitions currently run on a single GPU, the different partitions are more similar to virtual GPUs than individual workers, but the format of halo exchange is otherwise the same.

For workload simplicity, this demo framework uses a single-layer of message passing with tanh normalization, and a sigmoid activation layer. Because floating point addition is non-deterministic on the GPU, different runs with the same random seed can produce different results. Additionally, because of the simple GNN architecture that uses random feature, it tends to perform less well on larger graphs. The GNN uses max-cut as a workload. Max-cut is an NP-hard graph problem that partitions the vertices of a graph into two groups, such that the weight of the edges crossing between the groups is maximized.

## CUDA Implementation
The framework uses custom cuda kernels for performance. Kernels that involve traversing edges use warp-per-vertex, where every lane in the warp handles edges for more coalesced memory reads. Each lane stores a local accumulator, and the whole warp uses warp-shuffle primitives at the end to aggregate the answer. On kernels that involve getting projection gradients, to reduce atomic contention, threads first write to shared memory, before the whole block writes to global memory.

## Build Instructions

First, download NCCL
```
sudo apt install libnccl2 libnccl-dev
```

Next, run the build script

```
sudo chmod +x ./build.sh
./build.sh
```

## Running Instructions

The build script produces two executables. Example GPU training:
```
./gnn --iterations 100 --nodes 10000 --partitions 4
```

Example CPU training:
```
./cpu_gnn --iterations 100 --nodes 10000
```
## Performance
The following table compares per-iteration runtimes of single-partition GPU, 4-partition GPU, and CPU. Note that halo exchange on partitioned GPU is done with memcpyAsync rather than NCCL over PCIe. All times are per-iteration. Uses a power-law graph for training.

| Nodes  | Iterations| GPU Weight | Random Initialization Weight | GPU (ms) | Partitioned GPU(ms) | CPU (ms) | Single Partition Speedup | Multi Partition Speedup |
|--------|-----------|------------|------------------------------|----------|---------------------|----------|--------------------------|-------------------------|
|  30    |   100     |   135      |    89                        |  0.2     |    0.7              | 0.01     |     0.05x                |   0.015x                |
|10000   |   1000    |  30726     |   33780                      |  1.5     |    2.3              | 4.5      |       3x                 |    1.9x                 |
|30000   |  1000     |  80746     |  100745                      |  3.41    |    4.94             | 13.70    |      4x                  |     2.7x                |
|100000  |  1000     |  309512    |  336929                      |  8.96    |    11.79            | 45.13    |      5x                  |      3.8x               |

## Notes

### Future Updates
Currently planning on adding laplacian features, which will likely perform better than random features.

### Hardware Used
GPU: GeForce RTX 3060 CPU: AMD Ryzen 5 5600X
