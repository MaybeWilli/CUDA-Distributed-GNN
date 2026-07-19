#include "training_utils.cuh"

float training(GraphPartition& partition, int partitions, vector<GpuGnn>& gnns, vector<float>& total_probabilities, int num_devices, int iterations, 
    int biggest_buffer)
{
    //handle nccl variables
    ncclComm_t comms[num_devices];
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclGroupStart();

    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        ncclCommInitRank(
            &comms[i],
            num_devices,
            id,
            i
        );
    }

    ncclGroupEnd();

    int* request_ids[num_devices];
    int* request_locations[num_devices];
    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        checkCuda ( cudaMalloc((void**)&request_ids[i], biggest_buffer * sizeof(int)));
        checkCuda ( cudaMalloc((void**)&request_locations[i], biggest_buffer * sizeof(int)));
    }

    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    cudaEventRecord(gpu_start);
    for (int iter = 0; iter < iterations; iter++)
    {
        for (int i = 0; i < partitions; i++)
        {
            gnns[i].forward_run();
        }

        //save probabilities
        if (iter == iterations - 1)
        {
            int index = 0;
            for (int i = 0; i < partitions; i++)
            {
                vector<float> probabilities(gnns[i].nodes);
                checkCuda( cudaMemcpy(probabilities.data(), gnns[i].d_probabilities, gnns[i].nodes * sizeof(float), cudaMemcpyDeviceToHost) );

                for (int k = 0; k < probabilities.size(); k++)
                {
                    total_probabilities[index] = probabilities[k];
                    index++;
                }
            }
        }
        for (int i = 0; i < partitions; i++)
        {
            for (int j = 0; j < partitions; j++)
            {
                if (i != j)
                {
                    gnns[i].clear_buffer();
                    cudaSetDevice(gnns[j].device);
                    int request_size = gnns[i].buff_size;

                    checkCuda ( cudaMemcpy(request_ids[gnns[j].device], partition.request_ids[i].data(), request_size * sizeof(int), cudaMemcpyHostToDevice) );
                    checkCuda ( cudaMemcpy(request_locations[gnns[j].device], partition.request_locations[i].data(), request_size * sizeof(int), cudaMemcpyHostToDevice) );
                    gnns[j].request_probabilities(
                        request_ids[gnns[j].device], request_locations[gnns[j].device], partition.offsets[j], gnns[i].d_buffer, gnns[i].device, request_size, 
                        gnns[i].nodes, comms[gnns[j].device]);

                    gnns[i].save_to_buffer();
                }
            }
            gnns[i].flush_buffer();
        }

        for (int i = 0; i < partitions; i++)
        {
            gnns[i].prob_backprop();
            gnns[i].sigmoid_backprop();
            gnns[i].proj_backprop();
            gnns[i].tanh_backprop();
        }

        for (int i = 0; i < partitions; i++)
        {
            gnns[i].message_backprop();
        }

        vector<float> projection_gradients(FEATURE_DIM, 0);
        vector<float> weight_gradients(FEATURE_DIM*FEATURE_DIM, 0);
        std::fill(projection_gradients.begin(), projection_gradients.end(), 0);
        std::fill(weight_gradients.begin(), weight_gradients.end(), 0);

        for (int i = 0; i < partitions; i++)
        {
            vector<float> temp_proj_gradients(FEATURE_DIM, 0);
            vector<float> temp_weight_gradients(FEATURE_DIM*FEATURE_DIM, 0);

            cudaSetDevice(gnns[i].device);
            checkCuda( cudaMemcpy(temp_proj_gradients.data(), gnns[i].d_proj_gradients, FEATURE_DIM * sizeof(float), cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(temp_weight_gradients.data(), gnns[i].d_layer_weight_gradients, FEATURE_DIM * FEATURE_DIM * sizeof(float), cudaMemcpyDeviceToHost) );
            std::transform(
                projection_gradients.begin(),
                projection_gradients.end(),
                temp_proj_gradients.begin(),
                projection_gradients.begin(),
                std::plus<float>()
            );
            std::transform(
                weight_gradients.begin(),
                weight_gradients.end(),
                temp_weight_gradients.begin(),
                weight_gradients.begin(),
                std::plus<float>()
            );
        }

        for (int i = 0; i < partitions; i++)
        {
            gnns[i].update_weights(projection_gradients, weight_gradients);
        }
    }

    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, gpu_start, gpu_stop);
    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        checkCuda ( cudaFree(request_ids[i]));
        checkCuda ( cudaFree(request_locations[i]));
    }

    return milliseconds;
}