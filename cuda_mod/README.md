# cuda_mod

Módulo CUDA para aceleração do solver FEM climático

## Estrutura
- `CMakeLists.txt`: build do módulo compartilhado Python/CUDA
- `src/cuda_fem_solver.cu`: kernel CUDA para difusão de calor FEM
- `src/pybind_wrapper.cpp`: binding Python via pybind11

## Como usar
1. Compile com CMake (requer CUDA, pybind11, Python dev)
2. Importe em Python:
```python
import cuda_fem_solver
result = cuda_fem_solver.heat_diffusion(node_coords, connectivity, temp_in, kappa, dt)
```

## Integração
- O carregamento da malha (node_coords, connectivity) deve ser feito em Python e passado como numpy arrays para o módulo.
