__kernel void compute_radiation(__global const float* temp_in,
                                __global float* temp_out,
                                __global const float* surface_temp_in,
                                __global float* surface_temp_out,
                                __global const float* albedo,
                                __global const float* latitudes,
                                const float dt,
                                const int n_elements,
                                const int n_layers) {
    int gid = get_global_id(0);
    if (gid >= n_elements) return;

    // Constants
    const float SOLAR_CONSTANT = 1361.0f; // W/m^2
    const float STEFAN_BOLTZMANN = 5.67e-8f;
    const float CP_AIR = 1004.0f; // J/(kg K)
    const float AIR_MASS_PER_LAYER = 10000.0f; // kg/m^2 (approx)
    const float SURFACE_HEAT_CAPACITY = 1.0e6f; // J/(m^2 K) (approx for land/ocean mix)

    // 1. Surface Radiation Balance
    float T_surf = surface_temp_in[gid];
    float alb = albedo[gid];
    float lat = latitudes[gid];
    
    // Incoming Solar (Latitude dependent)
    // Average daily insolation approx: (S / PI) * cos(lat)
    // This creates the equator-to-pole temperature gradient driver
    float solar_in = (SOLAR_CONSTANT / 3.14159f) * cos(lat) * (1.0f - alb);
    if (solar_in < 0.0f) solar_in = 0.0f;
    
    // Outgoing Longwave (Blackbody)
    // Simple Greenhouse Effect approximation:
    // Reduce net outgoing longwave radiation to account for atmospheric back-radiation.
    // Earth's surface effectively loses only about 60% of blackbody radiation to space/atmosphere net.
    const float GREENHOUSE_FACTOR = 0.6f; 
    float lw_out = GREENHOUSE_FACTOR * STEFAN_BOLTZMANN * pown(T_surf, 4);
    
    // Net Flux
    float net_flux = solar_in - lw_out;
    
    // Update Surface Temp
    float dT_surf = (net_flux / SURFACE_HEAT_CAPACITY) * dt;
    surface_temp_out[gid] = T_surf + dT_surf;

    // 2. Atmospheric Cooling (Simplified Newtonian Cooling)
    // Relax towards a radiative equilibrium profile
    for(int l=0; l<n_layers; l++) {
        int idx = gid * n_layers + l;
        float T_air = temp_in[idx];
        
        // Target temp decreases with height
        float T_target = 288.15f - (l * 6.5f); 
        
        // Relaxation time scale (e.g., 10 days = 864000s)
        float tau = 864000.0f;
        
        float dT_air = -((T_air - T_target) / tau) * dt;
        
        // Also add some heating from surface (convection proxy)
        if (l == 0) {
            float T_diff = T_surf - T_air;
            if (T_diff > 0) {
                dT_air += (T_diff * 10.0f / tau) * dt; // Convective adjustment
            }
        }
        
        temp_out[idx] = T_air + dT_air;
    }
}
