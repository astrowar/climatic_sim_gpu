 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept> 
#include "climatic_data.hpp"
 

namespace py = pybind11;

//an class to estore mesh data  
class EarthMesh {
public:
    EarthMesh(const py::array_t<float>& node_coords,
              const py::array_t<int>& connectivity) {
        auto buf_coords = node_coords.request();
        auto buf_conn = connectivity.request();
        n_nodes = buf_coords.shape[0];
        n_elements = buf_conn.shape[0];
        buf_coords_ptr =    static_cast<const float*>(buf_coords.ptr);
        buf_conn_ptr=   static_cast<const int*>(buf_conn.ptr);

    }
 
    //remove copy constructor and assignment operator
    EarthMesh(const EarthMesh&) = delete;
    EarthMesh& operator=(const EarthMesh&) = delete;

    int n_nodes;
    int n_elements;
    const float*  buf_coords_ptr;
    const int*  buf_conn_ptr;

};

 extern "C" void alloc_memory(float** ptr, size_t size);
 extern "C" void alloc_memory_int(int** ptr, size_t size);
 extern "C" void copy_to_device(void* d_ptr, const void* h_ptr, size_t size);
 extern "C" void copy_to_host(void* h_ptr, const void* d_ptr, size_t size);

 
    ClimaticModelData::ClimaticModelData(EarthMesh* _mesh, int _numlayers) {
        mesh = _mesh;
 

        n_elements = _mesh->n_elements;
        n_layers = _numlayers;

        size_t full_size = n_elements * n_layers * sizeof(float);
        size_t surface_size = n_elements * sizeof(float);

        // Alocação (Idealmente usando cudaMallocManaged para facilitar)
        alloc_memory(&temp, full_size);
        alloc_memory(&q_vap, full_size);
        alloc_memory(&u_wind, full_size);
        alloc_memory(&v_wind, full_size);
        alloc_memory(&rho, full_size);
        alloc_memory(&pressure, full_size);

        alloc_memory(&surface_temp, surface_size);
        alloc_memory(&surface_albedo, surface_size);
        alloc_memory(&surface_type, surface_size);

        // Inicializar ponteiros CUDA para conectividade e coordenadas dos nós
        // Supondo que EarthMesh tenha os dados host: connectivity (int*) e node_coords (float*)
        size_t conn_size = n_elements * 4 * sizeof(int); // 4 nós por elemento (ajuste se necessário)
        size_t coords_size = _mesh->n_nodes * 3 * sizeof(float); // 3 coords por nó
 
        alloc_memory_int( &d_connectivity, conn_size);
        alloc_memory( &d_node_coords, coords_size);
        // Aqui, você deve garantir que _mesh->connectivity e _mesh->node_coords existam e estejam preenchidos   

        copy_to_device(d_connectivity, _mesh->buf_conn_ptr, conn_size);
        copy_to_device(d_node_coords, _mesh->buf_coords_ptr, coords_size);

     
    }
 



 
extern "C" void cuda_fem_solver_init();
extern "C" void ClimaticModel_init(ClimaticModelData* data, float temp0, float q_vap0, float u0, float v0, float rho0, float p0, float surf_temp0, float albedo0, float surf_type0);
extern "C" void ClimaticModel_step(ClimaticModelData* data, float dt);

static ClimaticModelData* climatic_data = nullptr;

//initialize mesh simulation
void cuda_simulation_init(py::array_t<float, py::array::c_style | py::array::forcecast> node_coords
    , py::array_t<int, py::array::c_style | py::array::forcecast> connectivity,
    int num_layers
) {
    
    EarthMesh* mesh  = new EarthMesh(node_coords, connectivity);
    climatic_data = new ClimaticModelData( mesh  , num_layers  );

    float temp_intial = 288.0f;      // Temperatura inicial (K)
    float q_vap_initial = 0.01f;     // Umidade específica inicial (kg/kg)
    float u_initial = 5.0f;          // Velocidade inicial U (m/s)
    float v_initial = 5.0f;          // Velocidade inicial V (m/s)
    float rho_initial = 1.225f;        // Densidade inicial (kg/m^3)
    float p_initial = 101325.0f;     // Pressão inicial (Pa)
    float surf_temp_initial = 288.0f; // Temperatura da superfície inicial (K)
    float albedo_initial = 0.3f;      // Albedo da superfície inicial
    float surf_type_initial = 1.0f;    // Tipo de superfície inicial (1: Terra) 
   ClimaticModel_init(climatic_data, temp_intial, q_vap_initial, u_initial, v_initial, rho_initial, p_initial, surf_temp_initial, albedo_initial, surf_type_initial);
   
}   


int  cuda_simulation_step(    
    float dt
) {
    if (climatic_data == nullptr) {
        throw std::runtime_error("Simulation not initialized. Call cuda_simulation_init first.");
    }
    ClimaticModel_step(climatic_data, dt); 
    return 0; // Return 0 on success
}


py::array_t<float> get_array(){
// Allocate and initialize some data; make this big so
        // we can see the impact on the process memory use:
        constexpr size_t size = 100*1000*1000;
        double *foo = new double[size];
        for (size_t i = 0; i < size; i++) {
            foo[i] = (double) i;
        }

        // Create a Python object that will free the allocated
        // memory when destroyed:
        py::capsule free_when_done(foo, [](void *f) {
            double *foo = reinterpret_cast<double *>(f);
            std::cerr << "Element [0] = " << foo[0] << "\n";
            std::cerr << "freeing memory @ " << f << "\n";
            delete[] foo;
        });

        return py::array_t<double>(
            {100, 1000, 1000}, // shape
            {1000*1000*8, 1000*8, 8}, // C-style contiguous strides for double
            foo, // the data pointer
            free_when_done); // numpy array references this parent

}


// Função para ler a temperatura superficial como array numpy contínuo

py::array_t<float> get_surface_temp() {
    if (climatic_data == nullptr) {
        throw std::runtime_error("Simulation not initialized. Call cuda_simulation_init first.");
    }
    int n_elem = climatic_data->n_elements;
    float* foo = new float[n_elem];
    copy_to_host(foo, climatic_data->surface_temp, n_elem * sizeof(float));

    //add some random to debug
    for (int i = 0; i < n_elem; i++) {
        foo[i] += 0.001f* static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/0.1f));
    }    

    py::capsule free_when_done(foo, [](void *f) {
        float *foo = reinterpret_cast<float *>(f);
        delete[] foo;
    });

    return py::array_t<float>(
        {n_elem}, // shape
        {sizeof(float)}, // stride
        foo, // data pointer
        free_when_done // capsule for memory management
    );
}

     
      



PYBIND11_MODULE(cuda_fem_solver, m) {
    m.def("init", &cuda_fem_solver_init, "Inicializa o contexto CUDA e reporta device");
    m.def("simulation_init", &cuda_simulation_init, "Inicializa a simulação climática com dados da malha");
    m.def("simulation_step", &cuda_simulation_step, "Executa um passo da simulação climática");
    m.def("get_surface_temp", &get_surface_temp, "Lê a temperatura superficial como array numpy contínuo");
}
