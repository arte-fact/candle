#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
    int num_gpus = 4;
    printf("=== C RCCL Test (%d GPUs) ===\n", num_gpus);

    ncclUniqueId id;
    ncclGetUniqueId(&id);

    // Init comms via threads (required: blocking collective)
    std::vector<ncclComm_t> comms(num_gpus);
    std::vector<hipStream_t> streams(num_gpus);
    std::vector<float*> sendbuff(num_gpus);
    std::vector<float*> recvbuff(num_gpus);

    std::vector<std::thread> threads;
    for (int g = 0; g < num_gpus; g++) {
        threads.emplace_back([&, g]() {
            hipSetDevice(g);
            hipStreamCreate(&streams[g]);
            ncclCommInitRank(&comms[g], num_gpus, id, g);
        });
    }
    for (auto& t : threads) t.join();
    printf("All comms initialized\n");

    // Allocate and fill
    for (int g = 0; g < num_gpus; g++) {
        hipSetDevice(g);
        hipMalloc(&sendbuff[g], 3 * sizeof(float));
        hipMalloc(&recvbuff[g], 3 * sizeof(float));
        float val = (g + 1);
        float data[3] = {val, val * 10, val * 100};
        hipMemcpy(sendbuff[g], data, 3 * sizeof(float), hipMemcpyHostToDevice);
        hipMemset(recvbuff[g], 0, 3 * sizeof(float));
        hipDeviceSynchronize();
        printf("GPU %d: src=[%.0f, %.0f, %.0f]\n", g, data[0], data[1], data[2]);
    }

    // AllReduce with group API
    printf("Launching AllReduce...\n");
    ncclGroupStart();
    for (int g = 0; g < num_gpus; g++) {
        ncclAllReduce(sendbuff[g], recvbuff[g], 3, ncclFloat, ncclSum, comms[g], streams[g]);
    }
    ncclGroupEnd();
    printf("Group end OK\n");

    // Sync and read
    for (int g = 0; g < num_gpus; g++) {
        hipSetDevice(g);
        hipStreamSynchronize(streams[g]);
        float result[3];
        hipMemcpy(result, recvbuff[g], 3 * sizeof(float), hipMemcpyDeviceToHost);
        printf("GPU %d: result=[%.0f, %.0f, %.0f]\n", g, result[0], result[1], result[2]);
    }

    // Cleanup
    for (int g = 0; g < num_gpus; g++) {
        hipFree(sendbuff[g]);
        hipFree(recvbuff[g]);
        ncclCommDestroy(comms[g]);
        hipStreamDestroy(streams[g]);
    }
    printf("PASSED\n");
    return 0;
}
