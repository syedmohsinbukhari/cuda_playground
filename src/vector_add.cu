#include<iostream>
#include<chrono>
#include<memory>
#include<string>
#include<cuda_runtime.h>


class TimeIt{
private:
    std::chrono::time_point<std::chrono::system_clock> start_time;

public:
    TimeIt();
    ~TimeIt();
};

TimeIt::TimeIt() {
    this->start_time = std::chrono::system_clock::now();
}

TimeIt::~TimeIt() {
    auto end_time = std::chrono::system_clock::now();

    auto duration_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - this->start_time
    ).count();

    std::cout << duration_time << " us" << std::endl;
}


__global__
void vector_add(const float *X, const float *Y, float *Z, int nEle) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < nEle) {
        Z[i] = X[i] + Y[i];
    }
}

float * initialize_host_vector(int nEle) {
    size_t mem_size = nEle * sizeof(float);

    float * h_V = (float *)malloc(mem_size);

    if (h_V == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    return h_V;
}

void randomize_host_vector(float * h_V, int nEle) {
    for (int i = 0; i < nEle; ++i) {
        h_V[i] = rand()/(float)RAND_MAX;
    }
}

float * initialize_random_host_vector(int nEle) {
    float * h_V = initialize_host_vector(nEle);
    randomize_host_vector(h_V, nEle);

    return h_V;
}

float * initialize_device_vector(int nEle) {
    size_t mem_size = nEle * sizeof(float);

    cudaError_t err = cudaSuccess;
    float * d_V = nullptr;
    err = cudaMalloc((void **)&d_V, mem_size);
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Failed to allocate device vector (error code %s)!\n",
            cudaGetErrorString(err)
        );
        exit(EXIT_FAILURE);
    }

    return d_V;
}

float * host_vector_to_device(float * h_V, int nEle) {
    size_t mem_size = nEle * sizeof(float);

    float * d_V = initialize_device_vector(nEle);

    cudaError_t err = cudaSuccess;
    {
        std::cout << "Host to device copy ";
        auto tmr = TimeIt();
        err = cudaMemcpy(d_V, h_V, mem_size, cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Failed to copy vector from host to device (error code %s)!\n",
            cudaGetErrorString(err)
        );
        exit(EXIT_FAILURE);
    }

    return d_V;
}

float * device_vector_to_host(float * d_V, int nEle) {
    size_t mem_size = nEle * sizeof(float);

    float * h_V = initialize_host_vector(nEle);

    cudaError_t err = cudaSuccess;
    {
        std::cout << "Device to host copy ";
        auto tmr = TimeIt();
        err = cudaMemcpy(h_V, d_V, mem_size, cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Failed to copy vector from device to host (error code %s)!\n",
            cudaGetErrorString(err)
        );
        exit(EXIT_FAILURE);
    }

    return h_V;
}

int getBlocksPerGrid(int nEle, int threadsPerBlock) {
    return (nEle + threadsPerBlock - 1) / threadsPerBlock;
}

float * add_device_vectors(
    const float * d_X, const float * d_Y, int nEle, int threadsPerBlock
) {
    float * d_Z = initialize_device_vector(nEle);
    int blocksPerGrid = getBlocksPerGrid(nEle, threadsPerBlock);
    {
        std::cout << "Vector addition on device ";
        auto tmr = TimeIt();
        vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, d_Z, nEle);
    }
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Failed to launch vector_add kernel (error code %s)!\n",
            cudaGetErrorString(err)
        );
        exit(EXIT_FAILURE);
    }

    return d_Z;
}

float * add_host_vectors(const float * h_X, const float * h_Y, int nEle) {
    float * hZ_ = initialize_host_vector(nEle);
    {
        std::cout << "Vector addition on host ";
        auto tmr = TimeIt();
        for (int i=0; i < nEle; i++) {
            hZ_[i] = h_X[i] + h_Y[i];
        }
    }
    return hZ_;
}

void freeCudaVector(float * d_V) {
    cudaError_t err = cudaSuccess;
    cudaFree(d_V);
    if (err != cudaSuccess) {
        fprintf(
            stderr,
            "Failed to free device vector (error code %s)!\n",
            cudaGetErrorString(err)
        );
        exit(EXIT_FAILURE);
    }
}

bool match_host_and_device_results(
    const float * h_Z, const float * hZ_, int nEle
) {
    bool verified = true;
    for (int i=0; i < nEle; i++) {
        if (fabs(h_Z[i] - hZ_[i]) > 1e-5) {
            verified = false;
        }
    }
    return verified;
}

void print_host_vector(const float * h_V, const int nEle, std::string mesg) {
    std::cout << mesg << ": ";

    for (int i=0; i < nEle; i++) {
        std::cout << h_V[i] << ",";
    }

    std::cout << std::endl;
}

int main(void) {
    std::cout << "Vector addition code using CUDA!" << std::endl;

    int nEle = 10000000;
    int threadsPerBlock = 256;

    float * h_X = initialize_random_host_vector(nEle);
    float * h_Y = initialize_random_host_vector(nEle);

    float * d_X = host_vector_to_device(h_X, nEle);
    float * d_Y = host_vector_to_device(h_Y, nEle);

    float * d_Z = add_device_vectors(d_X, d_Y, nEle, threadsPerBlock);

    float * h_Z = device_vector_to_host(d_Z, nEle);
    float * hZ_ = add_host_vectors(h_X, h_Y, nEle);

    bool verified = match_host_and_device_results(h_Z, hZ_, nEle);
    if (verified) {
        std::cout << "Results matched on host and device" << std::endl;
    } else {
        std::cerr << "Results did NOT match on host and device" << std::endl;
    }

#if defined(DEBUG)
    print_host_vector(h_X, nEle, "h_X");
    print_host_vector(h_Y, nEle, "h_Y");
    print_host_vector(h_Z, nEle, "h_Z");
    print_host_vector(hZ_, nEle, "hZ_");
#endif

    free(h_X);
    free(h_Y);
    free(h_Z);
    free(hZ_);

    freeCudaVector(d_X);
    freeCudaVector(d_Y);
    freeCudaVector(d_Z);

    return 0;
}
