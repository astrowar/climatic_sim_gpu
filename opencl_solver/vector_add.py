import pyopencl as cl
import numpy as np
import os

def main():
    # Configuração do contexto e fila
    # Tenta pegar a primeira plataforma e dispositivo disponível
    try:
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        print(f"Usando dispositivo: {device.name}")
    except Exception as e:
        print("Erro ao inicializar OpenCL. Verifique se você tem drivers OpenCL instalados.")
        print(e)
        return

    # Tamanho dos vetores (2D)
    width = 10
    height = 10
    
    # Criando dados de entrada (matrizes 2D)
    # Usamos float32 pois é o padrão comum para GPUs
    a_np = np.random.rand(height, width).astype(np.float32)
    b_np = np.random.rand(height, width).astype(np.float32)
    res_np = np.empty_like(a_np)

    # Criando buffers de memória no dispositivo
    mf = cl.mem_flags
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    res_buf = cl.Buffer(context, mf.WRITE_ONLY, res_np.nbytes)

    # Lendo o código do Kernel OpenCL de um arquivo externo
    kernel_path = os.path.join(os.path.dirname(__file__), 'kernels.cl')
    with open(kernel_path, 'r') as f:
        kernel_code = f.read()

    # Compilando o programa
    try:
        prg = cl.Program(context, kernel_code).build()
    except cl.RuntimeError as e:
        print("Erro ao compilar o kernel OpenCL:")
        print(e)
        return

    # Executando o kernel
    # Global size é uma tupla (width, height) para execução 2D
    # Passamos width como argumento para calcular o índice linear corretamente
    prg.sum_matrix(queue, (width, height), None, a_buf, b_buf, res_buf, np.int32(width))

    # Copiando o resultado de volta para o host
    cl.enqueue_copy(queue, res_np, res_buf)

    # Verificação
    print("\nMatriz A (primeira linha):")
    print(a_np[0])
    print("\nMatriz B (primeira linha):")
    print(b_np[0])
    print("\nResultado (A + B) (primeira linha):")
    print(res_np[0])

    # Verificando se o cálculo está correto usando numpy
    if np.allclose(res_np, a_np + b_np):
        print("\nSucesso: O cálculo OpenCL corresponde ao cálculo do Numpy!")
    else:
        print("\nErro: O cálculo OpenCL não corresponde!")

if __name__ == "__main__":
    main()
