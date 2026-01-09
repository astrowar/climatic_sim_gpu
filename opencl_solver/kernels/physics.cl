__kernel void compute_physics(__global const float* surface_temp,
                              __global float* surface_albedo,
                              const int n_elements) {
    int gid = get_global_id(0);
    if (gid >= n_elements) return;
    
    float T = surface_temp[gid];
    
    // Snow/Ice Albedo Feedback
    // Base albedo (Ocean/Land)
    float base_albedo = 0.3f;
    float snow_albedo = 0.55f;
    
    // Freezing point
    float T_freeze = 273.15f;
    float T_deep_freeze = 263.15f; // -10 C
    
    float current_albedo = base_albedo;
    
    if (T < T_deep_freeze) {
        // Full snow cover
        current_albedo = snow_albedo;
    } else if (T < T_freeze) {
        // Partial snow cover (linear transition)
        float fraction = (T_freeze - T) / (T_freeze - T_deep_freeze);
        current_albedo = base_albedo + fraction * (snow_albedo - base_albedo);
    }
    
    surface_albedo[gid] = current_albedo;
}
