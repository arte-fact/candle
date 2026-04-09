// Minimal test: compile this, embed it, load via hipModuleLoadData
#include <hip/hip_runtime.h>

extern "C" __global__ void test_add_one(
    const size_t numel,
    const float *inp,
    float *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = inp[i] + 1.0f;
    }
}
