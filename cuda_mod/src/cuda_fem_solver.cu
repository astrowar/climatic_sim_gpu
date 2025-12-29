#include <cuda_runtime.h>
#include "climatic_data.hpp"
#include <iostream>
#include <stdexcept>

// Kernel para inicializar arrays na GPU com valores arbitrários
__global__ void climaticmodel_init_kernel(
    float* temp, float temp0,
    float* q_vap, float q_vap0,
    float* u_wind, float u0,
    float* v_wind, float v0,
    float* rho, float rho0,
    float* pressure, float p0,
    int n_tot
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_tot) {
        temp[idx] = temp0;
        q_vap[idx] = q_vap0;
        u_wind[idx] = u0;
        v_wind[idx] = v0;
        rho[idx] = rho0;
        pressure[idx] = p0;
    }
}

__global__ void climaticmodel_init_surface_kernel(
    float* surface_temp, float surf_temp0,
    float* surface_albedo, float albedo0,
    float* surface_type, float surf_type0,
    int n_elem
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elem) {
        surface_temp[idx] = surf_temp0;
        surface_albedo[idx] = albedo0;
        surface_type[idx] = surf_type0;
    }
}

extern "C"
void ClimaticModel_init(ClimaticModelData* data, float temp0, float q_vap0, float u0, float v0, float rho0, float p0, float surf_temp0, float albedo0, float surf_type0) {
    int n_elem = data->n_elements;
    int n_layers = data->n_layers;
    int n_tot = n_elem * n_layers;
    int threads = 256;
    int blocks = (n_tot + threads - 1) / threads;
    climaticmodel_init_kernel<<<blocks, threads>>>(
        data->temp, temp0,
        data->q_vap, q_vap0,
        data->u_wind, u0,
        data->v_wind, v0,
        data->rho, rho0,
        data->pressure, p0,
        n_tot
    );
    cudaDeviceSynchronize();
    int blocks_surf = (n_elem + threads - 1) / threads;
    climaticmodel_init_surface_kernel<<<blocks_surf, threads>>>(
        data->surface_temp, surf_temp0,
        data->surface_albedo, albedo0,
        data->surface_type, surf_type0,
        n_elem
    );
    cudaDeviceSynchronize();
}
 

// Protótipos dos kernels (definidos em climatic_kernels.cu)
__global__ void fem_radiation_equilibrium_kernel(
    float* temp_air,
    float* surface_temp,
    const float* albedo,
    float* temp_air_out,
    float* surface_temp_out,
    int n_elements,
    int n_layers,
    float dt
);

__global__ void fem_dynamics_momentum_kernel(
    float* temp,
    float* pressure,
    float* u_wind,
    float* v_wind,
    float* u_wind_out,
    float* v_wind_out,
    int* d_connectivity,
    float* d_node_coords,
    int n_elements,
    int n_layers,
    float dt
);

__global__ void fem_advection_kernel(
    float* temp,
    float* temp_out,
    float* u_wind,
    float* v_wind,
    int* d_connectivity,
    int n_elements,
    int n_layers,
    float dt
);

__global__ void update_pressure_kernel(
    float* temp,
    float* pressure,
    int n_elements,
    int n_layers
);



extern "C" void cuda_fem_solver_init() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "[cuda_fem_solver] Nenhuma GPU CUDA encontrada!" << std::endl;
        throw std::runtime_error("No CUDA device found");
    }
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "[cuda_fem_solver] Usando GPU: " << prop.name << std::endl;
}
#include <iostream>

inline void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : " << cudaGetErrorString(result) << std::endl;
        throw std::runtime_error("CUDA failure");
    }
}



extern "C" void alloc_memory(float** ptr, size_t size) {
    //cuda memory allocation
    cudaMalloc((void**)ptr, size);
    checkCuda(cudaGetLastError(), "cudaMalloc failed");
}
 
extern "C" void alloc_memory_int(int** ptr, size_t size) {
    //cuda memory allocation
    cudaMalloc((void**)ptr, size);
    checkCuda(cudaGetLastError(), "cudaMalloc failed");
}

extern "C" void copy_to_host(float* dest, const float* src, size_t size) {
    checkCuda(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost), "cudaMemcpy to host failed");
}   

extern "C" void free_memory(float* ptr) {
    cudaFree(ptr);
}

extern "C" void copy_to_device(float* dest, const float* src, size_t size) {
    checkCuda(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice), "cudaMemcpy to device failed");
}

//header only kernels 
 

 

extern "C" void ClimaticModel_step(ClimaticModelData* data, float dt) {
    // Definir configuração de blocos (baseado em n_elements)
    int threadsPerBlock = 256;
    int blocksPerGrid = (data->n_elements + threadsPerBlock - 1) / threadsPerBlock;

    // --- PASSO 1: EQUILÍBRIO RADIATIVO (FÍSICA) ---
    // Calcula aquecimento solar e trocas de infravermelho
    // Aqui, os buffers "next" não existem na struct. Usar apenas os campos válidos.
    fem_radiation_equilibrium_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        data->temp, 
        data->surface_temp,
        data->surface_albedo,
        data->temp,     // sobrescreve temp diretamente
        data->surface_temp,
        data->n_elements,
        data->n_layers,
        dt
    );
    cudaDeviceSynchronize();

    // --- PASSO 2: DINÂMICA (MOMENTO E CORIOLIS) ---
    fem_dynamics_momentum_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        data->temp,
        data->pressure,
        data->u_wind,
        data->v_wind,
        data->u_wind, // sobrescreve u_wind diretamente
        data->v_wind, // sobrescreve v_wind diretamente
        data->d_connectivity,
        data->d_node_coords,
        data->n_elements,
        data->n_layers,
        dt
    );
    cudaDeviceSynchronize();

    // --- PASSO 3: ADVECÇÃO (TRANSPORTE) ---
    fem_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        data->temp,
        data->temp,
        data->u_wind,
        data->v_wind,
        data->d_connectivity,
        data->n_elements,
        data->n_layers,
        dt
    );
    cudaDeviceSynchronize();

    // --- PASSO 4: AJUSTE HIDROSTÁTICO / PRESSÃO ---
    update_pressure_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        data->temp,
        data->pressure,
        data->n_elements,
        data->n_layers
    );
    cudaDeviceSynchronize();
}