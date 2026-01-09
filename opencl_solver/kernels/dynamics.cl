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
        // REMOVED: Explicit 3-cell imposition.
        // REPLACED WITH: Simple approximation of PGF based on Temperature Gradient.
        
        // dP/dy approx proportional to dT/dy
        // T is high at equator, low at poles.
        // dT/dy is negative in NH (getting colder going North)
        // dT/dy is positive in SH (getting warmer going North towards Equator)
        
        // A simple parameterization of the meridional temperature gradient:
        // T ~ cos(lat)
        // dT/dlat ~ -sin(lat)
        
        // PGF force direction (High T -> Low T, High P -> Low P aloft)
        // Aloft: Flow from Equator to Pole.
        // Surface: Return flow (Pole to Equator).
        
        float sign_lat = (lat > 0.0f) ? 1.0f : -1.0f;
        
        // Natural tendency: Equator -> Pole aloft.
        // NH (Lat > 0): Equator is South. Pole is North. Flow North (+).
        // SH (Lat < 0): Equator is North. Pole is South. Flow South (-).
        // So poleward flow has sign = sign_lat.
        
        // PGF strength proportional to sin(2*lat) roughly (max at mid-lats, 0 at equator/poles)
        // or just proportional to latitude sine.
        
        // Layer factor: -1.0 at surface (return flow), +1.0 at top (poleward flow)
        float layer_factor = -1.0f + 2.0f * (float)l / (float)(n_layers - 1);
        
        // Force direction:
        // We want Poleward aloft (layer_factor > 0) -> +sign_lat
        // We want Equatorward surface (layer_factor < 0) -> -sign_lat
        
        float pgf_strength = 2.0e-4f; 
        // Note: This creates a single large Hadley-like cell per hemisphere.
        // The 3-cell structure (Ferrel/Polar) must emerge from Coriolis instability (baroclinic instability),
        // which requires higher resolution and better physics than this simple kernel.
        // But this removes the *forced* confinement.
        
        float F_pgf_y = pgf_strength * sign_lat * layer_factor;
        
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
        // REDUCED: Made timescale much longer (weak forcing) to allow dynamic eddies
        float nudging_timescale = 86400.0f * 5.0f; // 5 days (was 1 day)
        float F_forcing_u = (target_u - u) / nudging_timescale;

        // Coriolis Force
        float F_cx = f_coriolis * v;
        float F_cy = -f_coriolis * u;
        
        // Friction (Rayleigh damping)
        // INCREASED SLIGHTLY: From 1.0e-5 to 1.5e-5 to dampen excessive speeds
        float friction = (l == 0) ? 1.5e-5f : 5.0e-7f; 
        
        // Update Winds
        float du = (F_forcing_u + F_cx - friction * u) * dt;
        float dv = (F_pgf_y + F_cy - friction * v) * dt;
        
        u_wind_out[idx] = u + du;
        v_wind_out[idx] = v + dv;
    }
}
