__kernel void compute_dynamics(__global const float* temp_in,
                               __global const float* pressure,
                               __global const float* u_wind_in,
                               __global const float* v_wind_in,
                               __global float* u_wind_out,
                               __global float* v_wind_out,
                               __global const float* latitudes,
                               const float dt,
                               const int n_elements,
                               const int n_layers) {
    int gid = get_global_id(0);
    if (gid >= n_elements) return;

    // Constants
    const float R_EARTH = 6371000.0f;
    const float OMEGA = 7.2921e-5f; // Earth rotation rate (rad/s)
    const float RHO_AIR = 1.225f; // kg/m^3 (simplified constant for now)

    // Latitude from buffer
    float lat = latitudes[gid];
    float f_coriolis = 2.0f * OMEGA * sin(lat);

    for(int l=0; l<n_layers; l++) {
        int idx = gid * n_layers + l;
        
        float u = u_wind_in[idx];
        float v = v_wind_in[idx];
        
        // Simple Thermal Wind / Geostrophic approximation driver
        
        // 1. Meridional Pressure Gradient Force (PGF)
        // Driven by temperature difference between Equator (Hot) and Poles (Cold).
        // We want to simulate the 3-cell circulation model:
        // 0-30 (Hadley): Surface Equatorward, Aloft Poleward
        // 30-60 (Ferrel): Surface Poleward, Aloft Equatorward
        // 60-90 (Polar): Surface Equatorward, Aloft Poleward
        
        float abs_lat_deg = fabs(lat) * 180.0f / 3.14159f;
        float flow_sign = 0.0f;
        
        if (abs_lat_deg < 30.0f) {
            flow_sign = -1.0f; // Equatorward tendency
        } else if (abs_lat_deg < 60.0f) {
            flow_sign = 1.0f;  // Poleward tendency
        } else {
            flow_sign = -1.0f; // Equatorward tendency
        }
        
        float sign_lat = (lat > 0.0f) ? 1.0f : -1.0f;
        
        // Desired surface direction (North/South)
        // NH (lat>0): Equatorward is South (<0). (-1 * 1 = -1). OK.
        // SH (lat<0): Equatorward is North (>0). (-1 * -1 = +1). OK.
        float desired_surface_dir = flow_sign * sign_lat;
        
        // Layer factor: -1.0 at surface (l=0), +1.0 at top (l=n_layers-1)
        float layer_factor = -1.0f + 2.0f * (float)l / (float)(n_layers - 1);
        
        // Apply force
        // We want Surface (layer_factor = -1) to have force ~ desired_surface_dir
        // So we multiply by -layer_factor.
        // Surface: desired * -(-1) = desired.
        // Top: desired * -(1) = -desired (Return flow).
        
        float pgf_strength = 2.0e-4f; // Adjust magnitude
        float F_pgf_y = pgf_strength * desired_surface_dir * (-layer_factor);
        
        // 2. Zonal Nudging (Optional, to stabilize or enforce climatology)
        // We can relax this or make it layer dependent.
        // Surface: Easterlies (Trades). Aloft: Westerlies (Jet).
        // Let's reduce nudging strength and let Coriolis do more work, 
        // but keep some to prevent explosion.
        
        float target_u = 0.0f;
        float abs_lat = fabs(lat);
        const float PI = 3.14159f;
        
        // Simple profile: Easterly at surface, Westerly aloft
        if (layer_factor < 0.0f) {
             // Surface-ish
             if (abs_lat < 30.0f * PI / 180.0f) target_u = -5.0f; // Trades
             else target_u = 5.0f; // Mid-lat westerlies
        } else {
             // Aloft
             target_u = 15.0f; // Strong Westerlies (Jet) everywhere except maybe deep tropics
        }
        
        // Nudging towards climatological mean
        float nudging_timescale = 86400.0f; // 1 day
        float F_forcing_u = (target_u - u) / nudging_timescale;

        // Coriolis Force
        float F_cx = f_coriolis * v;
        float F_cy = -f_coriolis * u;
        
        // Friction (Rayleigh damping)
        // Stronger friction at surface
        float friction = (l == 0) ? 5.0e-5f : 1.0e-6f; 
        
        // Update Winds
        float du = (F_forcing_u + F_cx - friction * u) * dt;
        float dv = (F_pgf_y + F_cy - friction * v) * dt;
        
        u_wind_out[idx] = u + du;
        v_wind_out[idx] = v + dv;
    }
}
