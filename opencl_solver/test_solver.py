import sys
import os
import numpy as np

# Adiciona o diretório raiz ao path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from mesh_generator import generate_cubed_sphere_grid
    from opencl_solver import OpenCLFEMSolver
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    sys.exit(1)

def test_solver():
    print("=== Teste Básico do Solver OpenCL ===")
    
    # 1. Gerar uma malha simples (baixa resolução para teste rápido)
    print("Gerando malha esférica (5x5 por face)...")
    grid_points = generate_cubed_sphere_grid(n_points=5, radius=1.0)
    
    # 2. Inicializar o Solver
    print("Inicializando OpenCLFEMSolver...")
    try:
        solver = OpenCLFEMSolver(grid_points)
    except Exception as e:
        print(f"Falha na inicialização do solver: {e}")
        return

    print(f"Solver inicializado.")
    print(f"  Elementos: {solver.get_element_count()}")
    print(f"  Nós: {solver.get_node_count()}")

    # 3. Verificar estado inicial
    # O solver deve inicializar a temperatura da superfície em ~288.15K
    # Vamos pegar os valores iniciais interpolados
    print("\nVerificando estado inicial...")
    # update_simulation(0) apenas para pegar os valores atuais sem avançar muito (ou avançando 0)
    # Mas o método update_simulation roda um passo.
    # Vamos rodar um passo de tempo muito pequeno ou 0.
    initial_state = solver.update_simulation(dt=0.0)
    
    # Pegar temperatura da face 0
    temp_face0 = initial_state['scalars']['face_0']
    mean_temp = np.mean(temp_face0)
    print(f"  Temperatura média inicial (Face 0): {mean_temp:.4f} K")
    
    if np.abs(mean_temp - 288.15) < 1.0:
        print("  [OK] Temperatura inicial dentro do esperado.")
    else:
        print("  [FALHA] Temperatura inicial fora do esperado.")

    # 4. Rodar um passo de simulação
    dt = 60.0 # 60 segundos
    print(f"\nRodando simulação por {dt} segundos...")
    new_state = solver.update_simulation(dt=dt)
    
    temp_face0_new = new_state['scalars']['face_0']
    mean_temp_new = np.mean(temp_face0_new)
    print(f"  Nova temperatura média (Face 0): {mean_temp_new:.4f} K")
    
    # O kernel de radiação agora calcula balanço real (Solar - LW)
    # A temperatura pode subir ou descer dependendo da latitude e albedo
    # Não esperamos mais um aumento fixo linear
    
    diff = mean_temp_new - mean_temp
    
    print(f"  Variação observada: {diff:.6f} K")
    
    if abs(diff) > 0.0001:
        print("  [OK] A temperatura variou (simulação ativa).")
    else:
        print("  [FALHA] A temperatura permaneceu estática.")

    print("\nTeste concluído com sucesso!")

if __name__ == "__main__":
    test_solver()
