import numpy as np
import sys
import os

# Adicione o diretório do módulo compilado ao sys.path
#add ../build/lib to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),  '..', 'build', 'lib')))


try:
    import cuda_fem_solver
except ImportError as e:
    print(e)
    print("cuda_fem_solver não encontrado. Compile o módulo com CMake antes de rodar este exemplo.")
    exit(1)

# Inicializa contexto CUDA e reporta device
cuda_fem_solver.init()

# Exemplo: malha quadrada simples, NxN nodes e (N-1)x(N-1) elementos
dt =  100.0   # passo de tempo
n =10

nodes_pos = [ (x, y, 0) for x in range(n) for y in range(n) ]
node_coords = np.array(nodes_pos, dtype=np.float64)
connectivity = []
for i in range(n-1):
    for j in range(n-1):
        n0 = i*n + j
        n1 = n0 + 1
        n2 = n0 + n + 1
        n3 = n0 + n
        connectivity.append( [n0, n1, n2, n3] )
connectivity = np.array(connectivity, dtype=np.int32)

  
# LOGS DE DEBUG
print("node_coords.shape:", node_coords.shape)
print("connectivity.shape:", connectivity.shape)
 
print("n_nodes (from node_coords):", node_coords.shape[0])  
print("n_elements (from connectivity):", connectivity.shape[0])

# Chamada do solver CUDA
cuda_fem_solver.simulation_init(node_coords, connectivity,  5)
for cicles in range( 100):
    for loops in range(1000):
        cuda_fem_solver.simulation_step(dt)
    surf_temp = cuda_fem_solver.get_surface_temp( )
    result = np.array(surf_temp)[0]   
    print("Temperaturas após difusão:", result)
