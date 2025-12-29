// OpenCL Kernels for Climatic Simulation

__kernel void init_state(__global float* surface_temp, 
                         __global float* temp, 
                         const int n_elements, 
                         const int n_layers) {
    int gid = get_global_id(0);
    if (gid >= n_elements) return;
    
    // Initial surface temperature (e.g., 15 degrees Celsius)
    surface_temp[gid] = 288.15f; 
    
    // Initialize atmospheric profile with a simple lapse rate
    for(int l=0; l<n_layers; l++) {
        // Decrease temp with height (simplified)
        temp[gid * n_layers + l] = 288.15f - (l * 6.5f); 
    }
}

__kernel void simulation_step(__global float* surface_temp,
                              __global float* temp,
                              const float dt,
                              const int n_elements,
                              const int n_layers) {
    int gid = get_global_id(0);
    if (gid >= n_elements) return;
    
    // Very simple dummy physics:
    // Slowly rotate temperature pattern or just add noise/drift
    // For now, just a tiny drift to show change
    
    float current_temp = surface_temp[gid];
    
    // Simple relaxation towards a target or oscillation
    // float target = 288.15f + 10.0f * sin(gid * 0.01f);
    // surface_temp[gid] = current_temp + (target - current_temp) * 0.01f * dt;
    
    // Just add a small value for visualization test
    surface_temp[gid] = current_temp + 0.01f * dt;
}
