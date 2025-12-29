#ifndef CLIMATIC_DATA_HPP
#define CLIMATIC_DATA_HPP


class EarthMesh; // Forward declaration

 class ClimaticModelData {
public: 
    // 1. DIMENSÕES
    int n_elements;
    int n_layers;

    // 2. ESTADO ATMOSFÉRICO (n_elements * n_layers)
    // Organizado para Coalescência de Memória na GPU: [camada][elemento] ou [elemento][camada]
    float* temp;      // Temperatura (K)
    float* q_vap;     // Umidade específica (kg/kg) - ESSENCIAL para efeito estufa e nuvens
    float* u_wind;    // Velocidade horizontal U (m/s)
    float* v_wind;    // Velocidade horizontal V (m/s)
    float* rho;       // Densidade (kg/m^3)
    float* pressure;  // Pressão (Pa)

    // 3. ESTADO DA SUPERFÍCIE (n_elements)
    // A superfície aquece/resfria diferente do ar
    float* surface_temp;     
    float* surface_albedo;   
    float* surface_type;      // (0: Oceano, 1: Terra, 2: Gelo) - Muda a capacidade térmica!

    // 4. CAMPOS DE FLUXO (Opcional, para debug e balanço)
    float* flux_sw;           // Curto alcance (Solar)
    float* flux_lw;           // Longo alcance (Térmico)
    float* precipitation;     // Acúmulo de chuva


    EarthMesh* mesh;

    // Ponteiros CUDA para dados da malha
    int* d_connectivity;   // device pointer para conectividade
    float* d_node_coords;  // device pointer para coordenadas dos nós

    ClimaticModelData(EarthMesh* _mesh, int _numlayers);
};



#endif    