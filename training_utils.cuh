#ifndef TRAINING_UTILS_CUH
#define TRAINGIN_UTILS_CUH

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

float training(GraphPartition& partition, int partitions, vector<GpuGnn>& gnns, vector<float>& total_probabilities, int num_devices, int iterations, 
    int biggest_buffer);

#endif