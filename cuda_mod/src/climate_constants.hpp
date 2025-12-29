    
#ifndef CLIMATE_CONSTANTS_HPP
#define CLIMATE_CONSTANTS_HPP

// --- Constantes Físicas Universais ---
namespace Climate {
    // Constante de Stefan-Boltzmann (W m^-2 K^-4)
    const float SIGMA = 5.670374e-8f;
    
    // Constante Solar Média (W/m^2) no topo da atmosfera
    const float SOLAR_CONSTANT = 1361.0f;
    
    // Calor específico do ar seco a pressão constante (J / kg*K)
    const float CP_AIR = 1004.5f;
    
    // Massa total da coluna atmosférica por m^2 (kg/m^2) 
    // Baseado na pressão média de 1013.25 hPa (P/g)
    const float TOTAL_ATM_MASS_PER_M2 = 10332.0f;

    // Emissividade média de uma camada (para 3-6 camadas)
    // Nota: Em modelos simples, cada camada absorve uma fração da radiação.
    // Para 5 camadas, epsilon ~ 0.2 a 0.4 é comum para representar CO2/Vapor d'água.
    const float LAYER_EPSILON = 0.35f; 
    
    // Transmissividade (tau) = 1 - epsilon (assumindo reflexão desprezível no infravermelho)
    const float LAYER_TAU = 1.0f - LAYER_EPSILON;

    float R_dry = 287.05f;           // Constante dos gases para ar seco
    float g = 9.806f;                // Gravidade
    float L_v = 2.501e6f;            // Calor latente de vaporização
    float cp = 1004.5f;              // Calor específico

  
    
    // Absortividade Solar (Ondas Curtas): ~5% da energia solar é absorvida pelo ar
    const float LAYER_SW_ABS = 0.05f; 

}

#endif

  