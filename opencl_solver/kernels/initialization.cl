__kernel void init_state(__global float* surface_temp, 
                         __global float* temp, 
                         __global float* u_wind,
                         __global float* v_wind,
                         __global float* q_vap,
                         __global float* pressure,
                         __global float* rho,
                         __global float* surface_albedo,
                         __global int* surface_type,
                         __global const float* latitudes,
                         const int n_elements, 
                         const int n_layers) {
    int gid = get_global_id(0);
    if (gid >= n_elements) return;
    
    float lat = latitudes[gid];
    
    // Surface initialization with Latitude Gradient
    // Equator (0 deg): 305 K (32 C)
    // Pole (90 deg): 250 K (-23 C)
    // T = 305 - 55 * sin(lat)^2
    float T_surf = 305.0f - 55.0f * sin(lat) * sin(lat);
    
    surface_temp[gid] = T_surf; 
    surface_albedo[gid] = 0.3f;
    surface_type[gid] = 0; // Ocean by default
    
    // Atmospheric profile initialization
    const float P0 = 101325.0f; // Pa
    const float R = 287.05f;    // J/(kg K)
    const float g = 9.81f;      // m/s^2
    
    for(int l=0; l<n_layers; l++) {
        int idx = gid * n_layers + l;
        
        // Temperature: Decrease with height (simplified lapse rate)
        // Start from local surface temp
        float T = T_surf - (l * 6.5f);
        temp[idx] = T;
        
        // Wind: Start at rest
        u_wind[idx] = 0.0f;
        v_wind[idx] = 0.0f;
        
        // Humidity: Simple profile
        q_vap[idx] = 0.01f * (1.0f - (float)l/n_layers);
        
        // Pressure: Simplified hydrostatic (exponential decay approximation for height)
        // Assuming roughly 1km per layer for this simple test
        float height = l * 1000.0f;
        float P = P0 * exp(-g * height / (R * 288.15f));
        pressure[idx] = P;
        
        // Density: Ideal gas law
        rho[idx] = P / (R * T);
    }
}
