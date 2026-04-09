#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <cstdio>

int main() {
    // Init device
    hipSetDevice(0);
    printf("Device set\n");

    // Create stream
    hipStream_t stream;
    hipStreamCreate(&stream);
    printf("Stream created\n");

    // Allocate device memory
    float* d_data;
    hipMalloc(&d_data, 8 * sizeof(float));
    printf("Memory allocated\n");

    // Create generator
    hiprandGenerator_t gen;
    hiprandStatus_t st;
    st = hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_XORWOW);
    printf("Generator created: %d\n", st);

    st = hiprandSetStream(gen, stream);
    printf("Stream set: %d\n", st);

    st = hiprandSetPseudoRandomGeneratorSeed(gen, 42);
    printf("Seed set: %d\n", st);

    // Generate uniform (should work)
    st = hiprandGenerateUniform(gen, d_data, 8);
    printf("Uniform generated: %d\n", st);
    hipStreamSynchronize(stream);

    float h_data[8];
    hipMemcpy(h_data, d_data, 8 * sizeof(float), hipMemcpyDeviceToHost);
    printf("Uniform: %f %f %f %f\n", h_data[0], h_data[1], h_data[2], h_data[3]);

    // Generate normal (the one that crashes in candle)
    st = hiprandGenerateNormal(gen, d_data, 8, 0.0f, 1.0f);
    printf("Normal generated: %d\n", st);
    hipStreamSynchronize(stream);

    hipMemcpy(h_data, d_data, 8 * sizeof(float), hipMemcpyDeviceToHost);
    printf("Normal: %f %f %f %f\n", h_data[0], h_data[1], h_data[2], h_data[3]);

    hiprandDestroyGenerator(gen);
    hipFree(d_data);
    hipStreamDestroy(stream);
    printf("Done\n");
    return 0;
}
