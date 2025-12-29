#include <cuda_runtime.h>
#include "climate_constants.hpp"

// Kernel de equilíbrio radiativo
 __global__ void fem_radiation_equilibrium_kernel(
    float* temp_air,           // (n_elements * n_layers)
    float* surface_temp,       // (n_elements)
    const float* albedo,       // (n_elements)
    float* temp_air_out,       // (n_elements * n_layers)
    float* surface_temp_out,   // (n_elements)
    int n_elements,
    int n_layers,
    float dt
) {
    int e_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (e_idx >= n_elements) return;

    int offset = e_idx * n_layers;
    float mass_layer = Climate::TOTAL_ATM_MASS_PER_M2 / (float)n_layers;
    float heat_cap_air = mass_layer * Climate::CP_AIR;
    
    // Inércia térmica da superfície (ajuste conforme o tipo: terra vs oceano)
    // Terra: ~800,000 J/(m2 K) | Oceano: ~40,000,000 J/(m2 K)
    float heat_cap_surf = 1e6f; 

    // --- 1. RADIAÇÃO SOLAR (Shortwave) ---
    float flux_solar = Climate::SOLAR_CONSTANT / 4.0f;
    float layer_solar_absorption[6]; // max 6 camadas

    for (int l = n_layers - 1; l >= 0; l--) {
        // Cada camada absorve uma pequena fração do sol
        float absorbed = flux_solar * Climate::LAYER_SW_ABS;
        layer_solar_absorption[l] = absorbed;
        flux_solar -= absorbed; 
    }
    // O que sobra atinge o solo e é multiplicado pelo albedo
    float net_solar_surf = flux_solar * (1.0f - albedo[e_idx]);

    // --- 2. RADIAÇÃO TÉRMICA (Longwave) ---
    float net_energy_air[6] = {0.0f};
    
    // A) Fluxo Descendente (Atmosfera -> Solo)
    float f_down = 0.0f; 
    for (int l = n_layers - 1; l >= 0; l--) {
        float T = temp_air[offset + l];
        float emission = Climate::LAYER_EPSILON * Climate::SIGMA * powf(T, 4.0f);
        
        net_energy_air[l] += Climate::LAYER_EPSILON * f_down; // Absorve de cima
        net_energy_air[l] -= 2.0f * emission;                // Emite para cima e baixo
        
        f_down = (f_down * Climate::LAYER_TAU) + emission;
    }
    // No final do loop, f_down é a radiação atmosférica que atinge o solo (Back-radiation)

    // B) Fluxo Ascendente (Solo -> Espaço)
    float T_s = surface_temp[e_idx];
    float f_up = Climate::SIGMA * powf(T_s, 4.0f); // Emissão do solo (Corpo Negro)
    float surface_emission = f_up; // Guardar para o balanço do solo

    for (int l = 0; l < n_layers; l++) {
        float T = temp_air[offset + l];
        float emission = Climate::LAYER_EPSILON * Climate::SIGMA * powf(T, 4.0f);

        net_energy_air[l] += Climate::LAYER_EPSILON * f_up; // Absorve de baixo
        f_up = (f_up * Climate::LAYER_TAU) + emission;
    }

    // --- 3. ATUALIZAÇÃO DAS TEMPERATURAS ---
    
    // Solo
    float net_surf_energy = net_solar_surf + f_down - surface_emission;
    // (Opcional) Adicionar Troca Sensível: calor sobe do solo para a camada 0 se o solo estiver quente
    float sensible_heat = 10.0f * (T_s - temp_air[offset]); // Coeficiente de transferência simples
    net_surf_energy -= sensible_heat;

    surface_temp_out[e_idx] = T_s + (net_surf_energy / heat_cap_surf) * dt;

    // Atmosfera
    for (int l = 0; l < n_layers; l++) {
        float total_q = net_energy_air[l] + layer_solar_absorption[l];
        if (l == 0) total_q += sensible_heat; // Primeira camada recebe calor do solo por contato

        float dT = (total_q / heat_cap_air) * dt;
        temp_air_out[offset + l] = temp_air[offset + l] + dT;
    }
}
 


// Kernel de dinâmica dos ventos (momento e coriolis)
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
) {
    // TODO: Implementar kernel
}

// Kernel de advecção (transporte)
__global__ void fem_advection_kernel(
    float* temp,
    float* temp_out,
    float* u_wind,
    float* v_wind,
    int* d_connectivity,
    int n_elements,
    int n_layers,
    float dt
) {
    // TODO: Implementar kernel
}

// Kernel de ajuste hidrostático / pressão
__global__ void update_pressure_kernel(
    float* temp,
    float* pressure,
    int n_elements,
    int n_layers
) {
    // TODO: Implementar kernel
}
