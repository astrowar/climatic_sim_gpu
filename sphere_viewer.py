"""
OpenGL Sphere Viewer

This module provides the OpenGL viewer class for cubed-sphere grids
with FEM visualization support using VBOs (Vertex Buffer Objects).
"""

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
import multiprocessing
import copy
import time
import threading
import queue
import copy

def run_simulation_process(solver_class, grid_points, elevation_data, dt, data_queue, running_event):
    """Standalone process to run simulation without GIL interference."""
    print("[SIM PROCESS] Process started")
    
    try:
        # Re-initialize solver in this process
        # This ensures CUDA/C++ context is created in this process
        solver = solver_class(grid_points, elevation_data=elevation_data)
        print("[SIM PROCESS] Solver initialized")
    except Exception as e:
        print(f"[SIM PROCESS] Failed to initialize solver: {e}")
        return

    frame_count = 0
    iterations_per_update = 1
    step_count = 0
    
    while running_event.is_set():
        try:
            sim_result = solver.update_simulation(dt)
            step_count += 20
            
            if step_count >= iterations_per_update:
                step_count = 0
                try:
                    # Prepare data for queue
                    data_copy = {
                        'node_values': sim_result['scalars'],
                        'wind_values': sim_result['vectors'],
                        'pressure_values': sim_result['pressure'],
                        'vertical_motion_values': sim_result['vertical_motion'],
                        'time': solver.time,
                        'frame': frame_count
                    }
                    
                    # Put in queue (blocking with timeout to allow checking running_event)
                    if not data_queue.full():
                        data_queue.put(data_copy)
                        if frame_count % 5 == 0:
                            print(f"[SIM PROCESS] Sent frame {frame_count}")
                        frame_count += 1
                except Exception as e:
                    print(f"[SIM PROCESS] Error putting data: {e}")
            
            # Yield to avoid 100% CPU usage in this core
            time.sleep(0.01) 
            
        except Exception as e:
            print(f"[SIM PROCESS] Error in loop: {e}")
            break
            
    print("[SIM PROCESS] Process stopped")


class OpenGLSphereViewer:
    """Interactive OpenGL viewer for cubed-sphere grids."""
    
    def __init__(self, width: int = 1200, height: int = 900, title: str = "Cubed-Sphere Grid Viewer"):
        """Initialize the OpenGL viewer."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        self.width = width
        self.height = height
        self.window = glfw.create_window(width, height, title, None, None)
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        self._init_gl()
        
        # Camera parameters
        self.camera_distance = 5.0
        self.rotation_x = 20.0
        self.rotation_y = 45.0
        self.mouse_down = False
        self.last_mouse_pos = (0.0, 0.0)
        
        # Rendering options
        self.show_wireframe = False
        self.show_points = False
        self.show_faces = True
        self.show_wind = True  # Toggle for wind visualization
        self.scalar_mode = 'temperature' # 'temperature' or 'pressure'
        
        # Grid data
        self.grid_points = None
        self.face_colors = [
            (1.0, 0.0, 0.0),  # Red: +X
            (0.0, 0.0, 1.0),  # Blue: -X
            (0.0, 1.0, 0.0),  # Green: +Y
            (1.0, 1.0, 0.0),  # Yellow: -Y
            (0.0, 1.0, 1.0),  # Cyan: +Z
            (1.0, 0.0, 1.0),  # Magenta: -Z
        ]
        
        # FEM visualization data
        self.node_values = None
        self.wind_values = None  # Store wind vectors
        self.pressure_values = None # Store pressure values
        self.vertical_motion_values = None # Store vertical motion proxy
        self.show_fem_colors = False
        self.fem_solver = None
        self.animate_fem = False
        
        # Elevation data for base colors
        self.elevation_reader = None
        self.elevation_data = None
        self.use_elevation_colors = True
        self.show_temperature_only = False  # Toggle to show only temperature
        
        # VBO data
        self.vbos = {}  # VBOs for each face
        self.vbo_initialized = False
        
        # Shader program
        self.shader_program = None
        self.use_shaders = True
        
        # Threading for decoupled simulation
        self.simulation_thread = None
        self.data_queue = queue.Queue(maxsize=2)  # Small buffer
        self.simulation_running = False
        self.use_threaded_simulation = False
        self.simulation_dt = 100.0  # Default time step (seconds)
        
        # Set up callbacks
        glfw.set_window_user_pointer(self.window, self)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)
        
    def _init_gl(self):
        """Configure OpenGL settings."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        
        # Set clear color (background)
        glClearColor(0.1, 0.1, 0.15, 1.0)
        
        # Compile and link shaders
        self._compile_shaders()
        
        # Set up perspective
        self._setup_perspective()
    
    def _compile_shaders(self):
        """Compile and link vertex and fragment shaders."""
        # Vertex shader with Phong lighting
        vertex_shader_source = """
        #version 120
        
        varying vec3 fragNormal;
        varying vec3 fragPosition;
        varying vec3 fragColor;
        
        void main() {
            // Transform vertex position
            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            
            // Pass position in view space for lighting
            fragPosition = vec3(gl_ModelViewMatrix * gl_Vertex);
            
            // Transform normal to view space
            fragNormal = normalize(gl_NormalMatrix * gl_Normal);
            
            // Pass vertex color
            fragColor = gl_Color.rgb;
        }
        """
        
        # Fragment shader with Phong lighting
        fragment_shader_source = """
        #version 120
        
        varying vec3 fragNormal;
        varying vec3 fragPosition;
        varying vec3 fragColor;
        
        uniform vec3 lightPos;      // Light position in view space
        uniform vec3 lightAmbient;
        uniform vec3 lightDiffuse;
        uniform vec3 lightSpecular;
        uniform float shininess;
        
        void main() {
            // Ambient component
            vec3 ambient = lightAmbient * fragColor;
            
            // Diffuse component
            vec3 norm = normalize(fragNormal);
            vec3 lightDir = normalize(lightPos - fragPosition);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = lightDiffuse * diff * fragColor;
            
            // Specular component (Phong)
            vec3 viewDir = normalize(-fragPosition);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
            vec3 specular = lightSpecular * spec;
            
            // Combine lighting components
            vec3 result = ambient + diffuse + specular;
            gl_FragColor = vec4(result, 1.0);
        }
        """
        
        # Compile vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_source)
        glCompileShader(vertex_shader)
        
        # Check vertex shader compilation
        if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(vertex_shader).decode()
            raise RuntimeError(f"Vertex shader compilation failed:\n{error}")
        
        # Compile fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_source)
        glCompileShader(fragment_shader)
        
        # Check fragment shader compilation
        if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(fragment_shader).decode()
            raise RuntimeError(f"Fragment shader compilation failed:\n{error}")
        
        # Link shader program
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        
        # Check program linking
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program).decode()
            raise RuntimeError(f"Shader program linking failed:\n{error}")
        
        # Clean up shaders (they're now in the program)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        # Set uniform values
        glUseProgram(self.shader_program)
        
        # Light parameters (in view space)
        light_pos_loc = glGetUniformLocation(self.shader_program, "lightPos")
        light_ambient_loc = glGetUniformLocation(self.shader_program, "lightAmbient")
        light_diffuse_loc = glGetUniformLocation(self.shader_program, "lightDiffuse")
        light_specular_loc = glGetUniformLocation(self.shader_program, "lightSpecular")
        shininess_loc = glGetUniformLocation(self.shader_program, "shininess")
        
        glUniform3f(light_pos_loc, 5.0, 5.0, 5.0)
        glUniform3f(light_ambient_loc, 0.3, 0.3, 0.3)
        glUniform3f(light_diffuse_loc, 0.8, 0.8, 0.8)
        glUniform3f(light_specular_loc, 0.5, 0.5, 0.5)
        glUniform1f(shininess_loc, 32.0)
        
        glUseProgram(0)
        
        print("Shaders compiled and linked successfully")
    
    def _setup_perspective(self):
        """Set up the projection matrix."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
    
    @staticmethod
    def _key_callback(window, key, scancode, action, mods):
        """Handle keyboard input."""
        viewer = glfw.get_window_user_pointer(window)
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_W and action == glfw.PRESS:
            viewer.show_wireframe = not viewer.show_wireframe
            print(f"Wireframe: {'ON' if viewer.show_wireframe else 'OFF'}")
        elif key == glfw.KEY_P and action == glfw.PRESS:
            viewer.show_points = not viewer.show_points
            print(f"Points: {'ON' if viewer.show_points else 'OFF'}")
        elif key == glfw.KEY_F and action == glfw.PRESS:
            viewer.show_faces = not viewer.show_faces
            print(f"Filled faces: {'ON' if viewer.show_faces else 'OFF'}")
        elif key == glfw.KEY_R and action == glfw.PRESS:
            viewer.reset_view()
            print("View reset")
        elif key == glfw.KEY_C and action == glfw.PRESS:
            viewer.show_fem_colors = not viewer.show_fem_colors
            print(f"FEM colors: {'ON' if viewer.show_fem_colors else 'OFF'}")
        elif key == glfw.KEY_A and action == glfw.PRESS:
            viewer.animate_fem = not viewer.animate_fem
            print(f"FEM animation: {'ON' if viewer.animate_fem else 'OFF'}")
        elif key == glfw.KEY_E and action == glfw.PRESS:
            viewer.use_elevation_colors = not viewer.use_elevation_colors
            print(f"Elevation colors: {'ON' if viewer.use_elevation_colors else 'OFF'}")
        elif key == glfw.KEY_T and action == glfw.PRESS:
            viewer.show_temperature_only = not viewer.show_temperature_only
            print(f"Temperature only mode: {'ON' if viewer.show_temperature_only else 'OFF'}")
            # Force update
            if viewer.vbo_initialized:
                for face_id in range(6):
                    viewer._update_color_vbo(face_id)
        elif key == glfw.KEY_M and action == glfw.PRESS:
            if viewer.scalar_mode == 'temperature':
                viewer.scalar_mode = 'pressure'
            elif viewer.scalar_mode == 'pressure':
                viewer.scalar_mode = 'vertical'
            else:
                viewer.scalar_mode = 'temperature'
            print(f"Scalar Mode: {viewer.scalar_mode.upper()}")
            # Force update
            if viewer.vbo_initialized:
                for face_id in range(6):
                    viewer._update_color_vbo(face_id)
        elif key == glfw.KEY_V and action == glfw.PRESS:
            viewer.show_wind = not viewer.show_wind
            print(f"Wind vectors: {'ON' if viewer.show_wind else 'OFF'}")
    
    @staticmethod
    def _mouse_button_callback(window, button, action, mods):
        """Handle mouse button events."""
        viewer = glfw.get_window_user_pointer(window)
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                viewer.mouse_down = True
                viewer.last_mouse_pos = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                viewer.mouse_down = False
    
    @staticmethod
    def _cursor_pos_callback(window, xpos, ypos):
        """Handle mouse movement."""
        viewer = glfw.get_window_user_pointer(window)
        if viewer.mouse_down:
            dx = xpos - viewer.last_mouse_pos[0]
            dy = ypos - viewer.last_mouse_pos[1]
            
            viewer.rotation_y += dx * 0.5
            viewer.rotation_x += dy * 0.5
            
            viewer.last_mouse_pos = (xpos, ypos)
    
    @staticmethod
    def _scroll_callback(window, xoffset, yoffset):
        """Handle mouse scroll for zoom."""
        viewer = glfw.get_window_user_pointer(window)
        viewer.camera_distance = max(2.0, min(20.0, viewer.camera_distance - yoffset * 0.3))
    
    @staticmethod
    def _resize_callback(window, width, height):
        """Handle window resize."""
        viewer = glfw.get_window_user_pointer(window)
        viewer.width = width
        viewer.height = height
        glViewport(0, 0, width, height)
        viewer._setup_perspective()
        
    def load_grid_data(self, grid_points, node_values=None, fem_solver=None, use_threads=False, elevation_reader=None, simulation_dt=100.0):
        """Load cubed-sphere grid data and optional FEM values."""
        self.grid_points = grid_points
        self.node_values = node_values
        self.fem_solver = fem_solver
        self.use_threaded_simulation = use_threads
        self.elevation_reader = elevation_reader
        self.simulation_dt = simulation_dt
        
        # Extract elevation data from grid if available
        if grid_points and 'face_0' in grid_points and 'elevation' in grid_points['face_0']:
            self.elevation_data = {}
            for face_id in range(6):
                self.elevation_data[f'face_{face_id}'] = grid_points[f'face_{face_id}']['elevation']
            print("Elevation data loaded for visualization")
        
        if node_values is not None:
            self.show_fem_colors = True
        if fem_solver is not None:
            self.animate_fem = True
        
        # Initialize VBOs with the grid data
        self._initialize_vbos()
            
        # Start simulation thread if using threads
        if use_threads and fem_solver is not None:
            print(f"[MAIN] Starting simulation in THREADED mode")
            self.start_simulation_thread()
        else:
            print(f"[MAIN] Using DIRECT simulation mode (coupled with rendering)")
    
    def _initialize_vbos(self):
        """Initialize Vertex Buffer Objects for each face."""
        if self.grid_points is None:
            return
        
        print("Initializing VBOs...")
        
        for face_id in range(6):
            face_data = self.grid_points[f'face_{face_id}']
            xs = face_data['x']
            ys = face_data['y']
            zs = face_data['z']
            
            rows, cols = xs.shape
            
            # Build vertex, normal and index arrays
            vertices = []
            normals = []
            indices_faces = []
            indices_wireframe = []
            
            # Flatten grid to vertex array
            for i in range(rows):
                for j in range(cols):
                    x, y, z = xs[i, j], ys[i, j], zs[i, j]
                    vertices.extend([x, y, z])
                    
                    # Normal is just normalized position (sphere surface)
                    r = np.sqrt(x**2 + y**2 + z**2)
                    normals.extend([x/r, y/r, z/r])
            
            # Build quad indices for filled faces
            for i in range(rows - 1):
                for j in range(cols - 1):
                    # Vertex indices for this quad
                    v0 = i * cols + j
                    v1 = i * cols + (j + 1)
                    v2 = (i + 1) * cols + (j + 1)
                    v3 = (i + 1) * cols + j
                    
                    # Two triangles per quad
                    indices_faces.extend([v0, v1, v2, v0, v2, v3])
            
            # Build line indices for wireframe
            # Horizontal lines
            for i in range(rows):
                for j in range(cols - 1):
                    v0 = i * cols + j
                    v1 = i * cols + (j + 1)
                    indices_wireframe.extend([v0, v1])
            
            # Vertical lines
            for i in range(rows - 1):
                for j in range(cols):
                    v0 = i * cols + j
                    v1 = (i + 1) * cols + j
                    indices_wireframe.extend([v0, v1])
            
            # Convert to numpy arrays
            vertices = np.array(vertices, dtype=np.float32)
            normals = np.array(normals, dtype=np.float32)
            indices_faces = np.array(indices_faces, dtype=np.uint32)
            indices_wireframe = np.array(indices_wireframe, dtype=np.uint32)
            
            # Create VBOs
            vertex_vbo = vbo.VBO(vertices)
            normal_vbo = vbo.VBO(normals)
            face_ibo = vbo.VBO(indices_faces, target=GL_ELEMENT_ARRAY_BUFFER)
            wireframe_ibo = vbo.VBO(indices_wireframe, target=GL_ELEMENT_ARRAY_BUFFER)
            
            # Store VBO data
            self.vbos[f'face_{face_id}'] = {
                'vertex_vbo': vertex_vbo,
                'normal_vbo': normal_vbo,
                'face_ibo': face_ibo,
                'wireframe_ibo': wireframe_ibo,
                'n_face_indices': len(indices_faces),
                'n_wireframe_indices': len(indices_wireframe),
                'n_vertices': rows * cols,
                'rows': rows,
                'cols': cols
            }
        
        self.vbo_initialized = True
        print(f"VBOs initialized for {len(self.vbos)} faces")
    
    def _update_color_vbo(self, face_id):
        """Update color VBO for a face based on elevation (base) and FEM values (overlay)."""
        if not self.vbo_initialized:
            return None
        
        vbo_data = self.vbos[f'face_{face_id}']
        rows = vbo_data['rows']
        cols = vbo_data['cols']
        
        colors = []
        
        # Check if we have elevation data
        has_elevation = self.elevation_data is not None and self.use_elevation_colors
        
        # Use fixed temperature color bounds: -20°C .. +50°C
        # Auto-scaling disabled per user request.
        temp_min = -20.0
        temp_max = 50.0

        for i in range(rows):
            for j in range(cols):
                # Temperature-only mode: skip base colors and elevation
                if self.show_temperature_only:
                    face_key = f'face_{face_id}'
                    
                    # Select data source based on mode
                    data_source = None
                    if self.scalar_mode == 'temperature':
                        data_source = self.node_values
                    elif self.scalar_mode == 'pressure':
                        data_source = self.pressure_values

                    if self.show_fem_colors and data_source is not None and face_key in data_source:
                        node_vals = data_source[face_key]
                        sim_value = node_vals[i, j]
                        if self.scalar_mode == 'temperature':
                            # Normalize using precomputed temp_min/temp_max when available
                            if temp_min is not None and temp_max is not None and temp_max != temp_min:
                                norm = (sim_value - temp_min) / (temp_max - temp_min)
                            else:
                                norm = float(sim_value)
                            norm = float(np.clip(norm, 0.0, 1.0))
                            color = self.simulation_to_color(norm)
                        else:
                            color = self.pressure_to_color(sim_value)
                    else:
                        # No simulation, show gray
                        color = (0.5, 0.5, 0.5)
                else:
                    # Normal mode: base color + optional overlay
                    # Start with base color
                    if has_elevation:
                        elevation = self.elevation_data[f'face_{face_id}'][i, j]
                        base_color = self.elevation_to_color(elevation)
                    else:
                        base_color = self.face_colors[face_id]
                    
                    # Apply FEM overlay if active
                    if self.show_fem_colors:
                        # Check if face key exists (might be missing during initialization)
                        face_key = f'face_{face_id}'
                        
                        # Select data source based on mode
                        data_source = None
                        if self.scalar_mode == 'temperature':
                            data_source = self.node_values
                        elif self.scalar_mode == 'pressure':
                            data_source = self.pressure_values
                        elif self.scalar_mode == 'vertical':
                            data_source = self.vertical_motion_values
                            
                        if data_source is not None and face_key in data_source:
                            node_vals = data_source[face_key]
                            sim_value = node_vals[i, j]
                            
                            # Blend base color with simulation color
                            if self.scalar_mode == 'temperature':
                                if temp_min is not None and temp_max is not None and temp_max != temp_min:
                                    norm = (sim_value - temp_min) / (temp_max - temp_min)
                                else:
                                    norm = float(sim_value)
                                norm = float(np.clip(norm, 0.0, 1.0))
                                sim_color = self.simulation_to_color(norm)
                            elif self.scalar_mode == 'pressure':
                                sim_color = self.pressure_to_color(sim_value)
                            else: # vertical
                                sim_color = self.vertical_to_color(sim_value)
                            
                            # Mix: 60% base + 40% simulation
                            color = (
                                0.6 * base_color[0] + 0.4 * sim_color[0],
                                0.6 * base_color[1] + 0.4 * sim_color[1],
                                0.6 * base_color[2] + 0.4 * sim_color[2]
                            )
                        else:
                            color = base_color
                    else:
                        color = base_color
                
                colors.extend([color[0], color[1], color[2]])
        
        colors = np.array(colors, dtype=np.float32)
        
        # Create or update color VBO
        if 'color_vbo' in vbo_data:
            # Reuse existing VBO by updating its data
            color_vbo = vbo_data['color_vbo']
            color_vbo.set_array(colors)
            # Rebind to update GPU data
            color_vbo.bind()
            color_vbo.copy_data()
            color_vbo.unbind()
        else:
            # Create new VBO on first use
            vbo_data['color_vbo'] = vbo.VBO(colors)
        
        return vbo_data['color_vbo']
    
    def reset_view(self):
        """Reset camera to default position."""
        self.camera_distance = 5.0
        self.rotation_x = 20.0
        self.rotation_y = 45.0
    
    def simulation_worker(self):
        """Worker thread that runs FEM simulation independently."""
        print("[SIM THREAD] Simulation thread started")
         
        frame_count = 0
        iterations_per_update = 1  # Run 20 simulation steps before updating visualization
        step_count = 0
        dt = self.simulation_dt  # Use configurable time step
        print(f"[SIM THREAD] Using dt = {dt:.1f}s ({dt/60:.1f} minutes)")
        
        while self.simulation_running:
            # Run single simulation step
            sim_result = self.fem_solver.update_simulation(dt)
            step_count += 20
            
            # Only update queue after 20 steps
            if step_count >= iterations_per_update:
                step_count = 0
                
                # Try to put in queue (non-blocking)
                try:
                    data_copy = {
                        'node_values': copy.deepcopy(sim_result['scalars']),
                        'wind_values': copy.deepcopy(sim_result['vectors']),
                        'pressure_values': copy.deepcopy(sim_result['pressure']),
                        'vertical_motion_values': copy.deepcopy(sim_result['vertical_motion']),
                        'time': self.fem_solver.time,
                        'frame': frame_count
                    }
                    self.data_queue.put(data_copy, block=False)
                    if frame_count % 5 == 0:
                        print(f"[SIM THREAD] Sent frame {frame_count}, sim_time={self.fem_solver.time:.1f}s, queue_size={self.data_queue.qsize()}")
                    frame_count += 1
                except queue.Full:
                    print("[SIM THREAD] Data queue full, skipping update")
                    pass
            
            # Small sleep to avoid consuming 100% CPU
            time.sleep(0.1)
        
        print("[SIM THREAD] Simulation thread stopped")
    
    def start_simulation_thread(self):
        """Start the simulation worker process."""
        if hasattr(self, 'simulation_process') and self.simulation_process.is_alive():
            return

        print(f"[MAIN] Starting simulation in MULTIPROCESSING mode")
        self.simulation_running = True
        
        # Use spawn context for CUDA compatibility
        ctx = multiprocessing.get_context('spawn')
        self.data_queue = ctx.Queue(maxsize=2)
        self.simulation_running_event = ctx.Event()
        self.simulation_running_event.set()
        
        self.simulation_process = ctx.Process(
            target=run_simulation_process,
            args=(self.fem_solver.__class__, self.grid_points, self.elevation_data, self.simulation_dt, self.data_queue, self.simulation_running_event)
        )
        self.simulation_process.start()
        print("[MAIN] Simulation process launched")
    
    def stop_simulation_thread(self):
        """Stop the simulation worker process."""
        if hasattr(self, 'simulation_process') and self.simulation_process.is_alive():
            print("[MAIN] Stopping simulation process...")
            self.simulation_running_event.clear()
            self.simulation_process.join(timeout=2.0)
            if self.simulation_process.is_alive():
                self.simulation_process.terminate()
            print("[MAIN] Simulation process stopped")
    
    def elevation_to_color(self, elevation: float) -> tuple:
        """Convert elevation to base color (blue=ocean, green=land)."""
        if elevation < 0:
            # Water: blue tones
            return (0.1, 0.3, 0.7)  # Ocean blue
        else:
            # Land: green tones based on elevation
            # 0m = light green, 8000m = dark green
            normalized = elevation / 8000.0
            
            # Light green to dark green
            r = 0.2 - 0.15 * normalized  # Slight reduction in red
            g = 0.8 - 0.4 * normalized   # Green decreases with height
            b = 0.2 - 0.15 * normalized  # Slight reduction in blue
            
            return (r, g, b)
    
    def simulation_to_color(self, value: float) -> tuple:
        """Convert simulation value [0, 1] to color overlay (temperature-like)."""
        # Red-yellow gradient for temperature visualization
        # 0 = blue/cold, 0.5 = yellow/warm, 1 = red/hot
        if value < 0.5:
            # Blue to yellow
            t = value * 2.0
            r = t
            g = t
            b = 1.0 - t
        else:
            # Yellow to red
            t = (value - 0.5) * 2.0
            r = 1.0
            g = 1.0 - t
            b = 0.0
        
        return (r, g, b)
    
    def pressure_to_color(self, value: float) -> tuple:
        """Convert pressure value [0, 1] to color overlay."""
        # Pressure map: Low (0) = Purple/Blue, High (1) = Red/Orange
        # 0.5 = Neutral (Green/White)
        
        if value < 0.5:
            # Low pressure: Purple (0) to Cyan (0.5)
            t = value * 2.0
            r = 0.5 * (1.0 - t)
            g = t
            b = 1.0
        else:
            # High pressure: Cyan (0.5) to Red (1.0)
            t = (value - 0.5) * 2.0
            r = t
            g = 1.0 - 0.5 * t
            b = 1.0 - t
            
        return (r, g, b)

    def vertical_to_color(self, value: float) -> tuple:
        """Convert vertical motion [0, 1] to color overlay."""
        # 0.0 = Strong Downdraft (Blue)
        # 0.5 = Neutral (White/Grey)
        # 1.0 = Strong Updraft (Red)
        
        if value < 0.5:
            # Blue (0) to White (0.5)
            t = value * 2.0
            r = t
            g = t
            b = 1.0
        else:
            # White (0.5) to Red (1.0)
            t = (value - 0.5) * 2.0
            r = 1.0
            g = 1.0 - t
            b = 1.0 - t
            
        return (r, g, b)

    def value_to_color(self, value: float) -> tuple:
        """Convert scalar value [0, 1] to RGB color."""
        # Simple grayscale: 0=white (1,1,1), 1=black (0,0,0)
        gray = 1.0 - value
        return (gray, gray, gray)
        
    def _apply_camera_transform(self):
        """Apply camera rotation and zoom transformations."""
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -self.camera_distance)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        
    def _draw_axes(self, length=2.0):
        """Draw coordinate axes using VBOs."""
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        # Axes vertices
        axes_vertices = np.array([
            0.0, 0.0, 0.0,  length, 0.0, 0.0,  # X axis
            0.0, 0.0, 0.0,  0.0, length, 0.0,  # Y axis
            0.0, 0.0, 0.0,  0.0, 0.0, length   # Z axis
        ], dtype=np.float32)
        
        # Axes colors
        axes_colors = np.array([
            1.0, 0.0, 0.0,  1.0, 0.0, 0.0,  # Red for X
            0.0, 1.0, 0.0,  0.0, 1.0, 0.0,  # Green for Y
            0.0, 0.0, 1.0,  0.0, 0.0, 1.0   # Blue for Z
        ], dtype=np.float32)
        
        # Use VBOs for axes
        if not hasattr(self, '_axes_vbo'):
            self._axes_vbo = vbo.VBO(axes_vertices)
            self._axes_color_vbo = vbo.VBO(axes_colors)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        self._axes_vbo.bind()
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        self._axes_color_vbo.bind()
        glColorPointer(3, GL_FLOAT, 0, None)
        
        glDrawArrays(GL_LINES, 0, 6)
        
        self._axes_vbo.unbind()
        self._axes_color_vbo.unbind()
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
    
    def _draw_face_filled(self, face_id, alpha=0.3):
        """Draw filled face using VBOs."""
        if not self.vbo_initialized:
            return
        
        vbo_data = self.vbos[f'face_{face_id}']
        
        # Use shader program if available
        if self.use_shaders and self.shader_program:
            glUseProgram(self.shader_program)
        
        # Update colors based on FEM values
        color_vbo = self._update_color_vbo(face_id)
        
        # Enable vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        # Bind and set vertex data
        vbo_data['vertex_vbo'].bind()
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        # Bind and set normal data
        vbo_data['normal_vbo'].bind()
        glNormalPointer(GL_FLOAT, 0, None)
        
        # Bind and set color data
        color_vbo.bind()
        glColorPointer(3, GL_FLOAT, 0, None)
        
        # Bind index buffer and draw
        vbo_data['face_ibo'].bind()
        glDrawElements(GL_TRIANGLES, vbo_data['n_face_indices'], GL_UNSIGNED_INT, None)
        
        # Unbind
        vbo_data['vertex_vbo'].unbind()
        vbo_data['normal_vbo'].unbind()
        color_vbo.unbind()
        vbo_data['face_ibo'].unbind()
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
        # Disable shader program
        if self.use_shaders and self.shader_program:
            glUseProgram(0)
    
    def _draw_face_wireframe(self, face_id):
        """Draw wireframe using VBOs."""
        if not self.vbo_initialized:
            return
        
        vbo_data = self.vbos[f'face_{face_id}']
        color = self.face_colors[face_id]
        
        glDisable(GL_LIGHTING)
        glColor3f(color[0], color[1], color[2])
        glLineWidth(1.5)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        
        vbo_data['vertex_vbo'].bind()
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        vbo_data['wireframe_ibo'].bind()
        glDrawElements(GL_LINES, vbo_data['n_wireframe_indices'], GL_UNSIGNED_INT, None)
        
        vbo_data['vertex_vbo'].unbind()
        vbo_data['wireframe_ibo'].unbind()
        
        glDisableClientState(GL_VERTEX_ARRAY)
    
    def _draw_face_points(self, face_id):
        """Draw grid points using VBOs."""
        if not self.vbo_initialized:
            return
        
        vbo_data = self.vbos[f'face_{face_id}']
        color = self.face_colors[face_id]
        
        glDisable(GL_LIGHTING)
        glColor3f(color[0], color[1], color[2])
        glPointSize(5.0)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        
        vbo_data['vertex_vbo'].bind()
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        glDrawArrays(GL_POINTS, 0, vbo_data['n_vertices'])
        
        vbo_data['vertex_vbo'].unbind()
        
        glDisableClientState(GL_VERTEX_ARRAY)
    
    def _draw_latitude_lines(self, radius=1.05, num_lines=5):
        """Draw latitude lines (parallels) around the sphere."""
        glDisable(GL_LIGHTING)
        glColor3f(0.8, 0.8, 0.2)  # Yellow-ish color
        glLineWidth(2.0)
        
        # Draw latitude lines at different angles
        num_segments = 64
        for lat_idx in range(num_lines):
            # Latitude angle from -75° to +75° (skip poles)
            lat = -75.0 + (150.0 / (num_lines - 1)) * lat_idx
            lat_rad = np.radians(lat)
            
            vertices = []
            y = radius * np.sin(lat_rad)
            r = radius * np.cos(lat_rad)
            
            for i in range(num_segments + 1):
                angle = 2.0 * np.pi * i / num_segments
                x = r * np.cos(angle)
                z = r * np.sin(angle)
                vertices.extend([x, y, z])
            
            vertices_array = np.array(vertices, dtype=np.float32)
            
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, vertices_array)
            glDrawArrays(GL_LINE_STRIP, 0, num_segments + 1)
            glDisableClientState(GL_VERTEX_ARRAY)
    
    def _draw_meridian_line(self, radius=1.05):
        """Draw meridian line from south pole to north pole."""
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.8, 0.8)  # Cyan color
        glLineWidth(2.5)
        
        num_segments = 64
        vertices = []
        
        # Draw a semicircle from south to north pole along prime meridian
        for i in range(num_segments + 1):
            # Angle from -90° (south) to +90° (north)
            angle = -np.pi/2 + np.pi * i / num_segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0.0
            vertices.extend([x, y, z])
        
        vertices_array = np.array(vertices, dtype=np.float32)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices_array)
        glDrawArrays(GL_LINE_STRIP, 0, num_segments + 1)
        glDisableClientState(GL_VERTEX_ARRAY)
    
    def _draw_north_marker(self, radius=1.35):
        """Draw letter 'N' at north pole using lines."""
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)  # White color
        glLineWidth(3.0)
        
        # Position at north pole
        y_pos = radius
        
        # Define N shape with lines (scaled appropriately)
        scale = 0.15
        x_offset = 0.0
        z_offset = 0.0
        
        # N consists of: left vertical, diagonal, right vertical
        n_vertices = np.array([
            # Left vertical line
            x_offset - scale*0.5, y_pos - scale, z_offset,
            x_offset - scale*0.5, y_pos + scale, z_offset,
            # Diagonal line
            x_offset - scale*0.5, y_pos + scale, z_offset,
            x_offset + scale*0.5, y_pos - scale, z_offset,
            # Right vertical line
            x_offset + scale*0.5, y_pos - scale, z_offset,
            x_offset + scale*0.5, y_pos + scale, z_offset,
        ], dtype=np.float32)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, n_vertices)
        glDrawArrays(GL_LINES, 0, 6)
        glDisableClientState(GL_VERTEX_ARRAY)
    
    def _draw_grid(self):
        """Draw the complete cubed-sphere grid."""
        if self.grid_points is None:
            return
        
        for face_id in range(6):           
            if self.show_faces:
                self._draw_face_filled(face_id, alpha=0.3)
            if self.show_wireframe:
                self._draw_face_wireframe(face_id)
            if self.show_points:
                self._draw_face_points(face_id)
        
        # Draw planetary features
        self._draw_latitude_lines()
        self._draw_meridian_line()
        self._draw_north_marker()
    
    def cleanup(self):
        """Clean up VBO resources to prevent memory leaks."""
        if self.vbo_initialized:
            print("Cleaning up VBOs...")
            for face_id in range(6):
                face_key = f'face_{face_id}'
                if face_key in self.vbos:
                    vbo_data = self.vbos[face_key]
                    
                    # Delete all VBOs for this face
                    if 'vertex_vbo' in vbo_data:
                        vbo_data['vertex_vbo'].delete()
                    if 'normal_vbo' in vbo_data:
                        vbo_data['normal_vbo'].delete()
                    if 'color_vbo' in vbo_data:
                        vbo_data['color_vbo'].delete()
                    if 'face_ibo' in vbo_data:
                        vbo_data['face_ibo'].delete()
                    if 'wireframe_ibo' in vbo_data:
                        vbo_data['wireframe_ibo'].delete()
            
            # Clean up axes VBOs if they exist
            if hasattr(self, '_axes_vbo'):
                self._axes_vbo.delete()
            if hasattr(self, '_axes_color_vbo'):
                self._axes_color_vbo.delete()
            
            self.vbos.clear()
            self.vbo_initialized = False
            print("VBOs cleaned up")
        
        # Clean up shader program
        if self.shader_program:
            glDeleteProgram(self.shader_program)
            self.shader_program = None
            print("Shader program deleted")
    
    def draw(self):
        """Render the scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._apply_camera_transform()
        self._draw_axes()
        self._draw_grid()
        #if self.show_wind:
        #    self._draw_wind_vectors()
        glfw.swap_buffers(self.window)
    
    def _draw_wind_vectors(self):
        """Draw wind vectors at two levels using spatially distributed sampling."""
        if self.wind_values is None or not self.vbo_initialized:
            return
            
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        # Get global wind data
        global_winds = self.wind_values
        
        # Get global nodes
        if self.fem_solver:
            global_nodes = self.fem_solver.global_nodes
        else:
            return
        
        # Calculate max speed for scaling
        max_speed = np.max(np.linalg.norm(global_winds, axis=1))
        target_max_len = 0.07
        scale = target_max_len / max_speed if max_speed > 0.001 else 1.0
        
        # Spatial sampling: select nodes that are sufficiently far apart
        # This ensures uniform density across the sphere
        if not hasattr(self, '_arrow_sources') or len(self._arrow_sources) == 0:
            print("[WIND] Computing spatially distributed arrow sources...")
            min_distance = 0.15  # Minimum distance between arrows (adjust for density)
            arrow_sources = []
            
            for node_id in range(len(global_nodes)):
                pos = global_nodes[node_id]
                
                # Check distance to all previously selected sources
                too_close = False
                for prev_id in arrow_sources:
                    prev_pos = global_nodes[prev_id]
                    dist = np.linalg.norm(pos - prev_pos)
                    if dist < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    arrow_sources.append(node_id)
            
            self._arrow_sources = arrow_sources
            print(f"[WIND] Selected {len(arrow_sources)} arrow sources from {len(global_nodes)} nodes")
        
        glBegin(GL_LINES)
        
        for node_id in self._arrow_sources:
            x, y, z = global_nodes[node_id]
            vx, vy, vz = global_winds[node_id]
            
            speed = np.sqrt(vx*vx + vy*vy + vz*vz)
            if speed < 0.001:
                continue
                
            mag = np.sqrt(x*x + y*y + z*z)
            nx, ny, nz = x/mag, y/mag, z/mag
            
            # --- LEVEL 1: SURFACE WINDS (White) ---
            glColor3f(1.0, 1.0, 1.0)
            self._draw_arrow(x, y, z, vx, vy, vz, scale, nx, ny, nz)
            
            # --- LEVEL 2: UPPER WINDS (Color coded by vertical motion) ---
            h = 1.1
            ux, uy, uz = x*h, y*h, z*h
            
            # Color based on vertical motion if available
            if self.vertical_motion_values is not None and self.fem_solver is not None:
                v_val = self.fem_solver.fluid_solver.vertical_velocity[node_id]
                v_mean = np.mean(self.fem_solver.fluid_solver.vertical_velocity)
                v_std = np.std(self.fem_solver.fluid_solver.vertical_velocity)
                if v_std > 1e-8:
                    v_normalized = (v_val - (v_mean - 2*v_std)) / (4*v_std)
                    v_normalized = np.clip(v_normalized, 0.0, 1.0)
                else:
                    v_normalized = 0.5
                glColor3f(v_normalized, 1.0 - v_normalized, 1.0 - v_normalized * 0.5)
            else:
                glColor3f(0.4, 0.7, 1.0)
                
            # Draw upper wind (reversed to show return flow)
            self._draw_arrow(ux, uy, uz, -vx, -vy, -vz, scale, nx, ny, nz)
                    
        glEnd()

    def _draw_arrow(self, x, y, z, vx, vy, vz, scale, nx, ny, nz):
        """Helper to draw a single arrow with fins that always face the camera."""
        xe, ye, ze = x + vx*scale, y + vy*scale, z + vz*scale
        
        # Main line
        glVertex3f(x, y, z)
        glVertex3f(xe, ye, ze)
        
        # Arrowhead
        dx, dy, dz = vx*scale, vy*scale, vz*scale
        
        # To make the arrowheads always visible regardless of camera angle,
        # we use the vector from the point to the camera (approximate)
        # or simply use the normal vector but add a second perpendicular wing.
        
        arrow_len = 0.4
        bx, by, bz = -dx * arrow_len, -dy * arrow_len, -dz * arrow_len
        
        speed = np.sqrt(vx*vx + vy*vy + vz*vz)
        ss = speed * scale * arrow_len * 0.5
        
        # Wing 1: Perpendicular to surface (Normal vector)
        sx1, sy1, sz1 = nx * ss, ny * ss, nz * ss
        
        # Wing 2: Tangent to surface (Cross product of wind and normal)
        # This ensures that even if the camera is looking straight down the normal,
        # the side wings are visible.
        tx = dy*nz - dz*ny
        ty = dz*nx - dx*nz
        tz = dx*ny - dy*nx
        t_mag = np.sqrt(tx*tx + ty*ty + tz*tz)
        
        if t_mag > 1e-10:
            sx2, sy2, sz2 = tx/t_mag * ss, ty/t_mag * ss, tz/t_mag * ss
        else:
            sx2, sy2, sz2 = 0, 0, 0

        # Draw 4 wings (cross shape) for maximum visibility from any angle
        glVertex3f(xe, ye, ze)
        glVertex3f(xe + bx + sx1, ye + by + sy1, ze + bz + sz1)
        glVertex3f(xe, ye, ze)
        glVertex3f(xe + bx - sx1, ye + by - sy1, ze + bz - sz1)
        
        glVertex3f(xe, ye, ze)
        glVertex3f(xe + bx + sx2, ye + by + sy2, ze + bz + sz2)
        glVertex3f(xe, ye, ze)
        glVertex3f(xe + bx - sx2, ye + by - sy2, ze + bz - sz2)
    
    def run(self):
        """Main rendering loop."""
        print("\n" + "="*60)
        print("OpenGL Cubed-Sphere Viewer (VBO + Shaders)")
        print("="*60)
        print("\nControls:")
        print("  Mouse drag       - Rotate view")
        print("  Mouse wheel      - Zoom in/out")
        print("  W                - Toggle wireframe")
        print("  P                - Toggle points")
        print("  F                - Toggle filled faces")
        print("  C                - Toggle FEM colors overlay")
        print("  E                - Toggle elevation base colors")
        print("  T                - Toggle temperature-only mode")
        print("  M                - Toggle scalar mode (Temp/Pressure/Vertical)")
        print("  V                - Toggle wind vectors")
        print("  A                - Toggle FEM animation")
        print("  R                - Reset view")
        print("  ESC              - Exit")
        if self.use_threaded_simulation:
            print("\n[MODE] Threaded simulation (decoupled)")
        else:
            print("\n[MODE] Direct simulation (coupled)")
        print("="*60 + "\n")
        
        last_time = time.time()
        frame_count = 0
        fps_time = last_time
        last_sim_frame = -1
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            if self.animate_fem and self.fem_solver is not None:
                if self.use_threaded_simulation:
                    try:
                        data = self.data_queue.get(block=False)
                        self.node_values = data['node_values']
                        if 'wind_values' in data:
                            self.wind_values = data['wind_values']
                        if 'pressure_values' in data:
                            self.pressure_values = data['pressure_values']
                        if 'vertical_motion_values' in data:
                            self.vertical_motion_values = data['vertical_motion_values']
                        sim_time = data['time']
                        sim_frame = data['frame']
                        
                        if sim_frame != last_sim_frame:
                            last_sim_frame = sim_frame
                            if sim_frame % 5 == 0:
                                print(f"[VIEWER] Received sim_frame={sim_frame}, sim_time={sim_time:.1f}s, render_fps={frame_count}")
                    except queue.Empty:
                        pass
                else:
                    sim_result = self.fem_solver.update_simulation(dt)
                    self.node_values = sim_result['scalars']
                    self.wind_values = sim_result['vectors']
                    self.pressure_values = sim_result['pressure']
            
            glfw.poll_events()
            self.draw()
            
            frame_count += 1
            if current_time - fps_time >= 1.0:
                if not self.use_threaded_simulation and self.animate_fem and self.fem_solver is not None:
                    #print(f"[VIS] FPS: {frame_count} | Time: {self.fem_solver.time:.2f}s")
                    pass
                elif self.use_threaded_simulation:
                    pass
                #print(f"[VIS] Render FPS: {frame_count}")
                frame_count = 0
                fps_time = current_time
            
            time.sleep(0.001)
        
        if self.use_threaded_simulation:
            self.stop_simulation_thread()
        
        # Clean up VBO resources before terminating
        self.cleanup()
        
        glfw.terminate()
