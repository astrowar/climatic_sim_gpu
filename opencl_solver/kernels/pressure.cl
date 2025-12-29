__kernel void compute_pressure(__global const float* temp,
                               __global float* pressure,
                               const int n_elements,
                               const int n_layers) {
    int gid = get_global_id(0);
    if (gid >= n_elements) return;

    const float P0 = 101325.0f;
    const float R = 287.05f;
    const float g = 9.81f;

    // Hydrostatic Integration from top down or bottom up
    // Simplified: Re-calculate based on height and local temp
    
    for(int l=0; l<n_layers; l++) {
        int idx = gid * n_layers + l;
        float T = temp[idx];
        float height = l * 1000.0f; // 1km layers
        
        // Hypsometric equation approximation
        pressure[idx] = P0 * exp(-g * height / (R * T));
    }
}
