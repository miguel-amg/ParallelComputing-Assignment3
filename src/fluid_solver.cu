#include "fluid_solver.h"
#include <cmath>
#include <cstring>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>

#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <chrono>


#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x) \
    { float *tmp = x0; x0 = x; x = tmp; }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20


// ----------------------------------------------------------------------------------------------------------------------------------------------

__global__ void add_source_kernel(int size, float *x, const float *s, float dt) {
    // Memória shared para os arrays s e x
    __shared__ float s_x[256];  
    __shared__ float s_s[256];  

    // Calculo de indices
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Indice global 
    int local_idx = threadIdx.x; // Indice local

    // Carregar dados para a memória shared
    if (i < size) {
        s_x[local_idx] = x[i];
        s_s[local_idx] = s[i];
    }
    __syncthreads(); // Sincronizar threads do bloco

    // Atualizar os do array na memoria shared
    if (i < size) {
        s_x[local_idx] += dt * s_s[local_idx];
    }
    __syncthreads(); // Sincronizar threads do bloco

    // Escrever os resultados na memoria global
    if (i < size) {
        x[i] = s_x[local_idx];
    }
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2); // LINHA ORIGINAL

    // Alocar memória na GPU para os dois arrays 
    float *d_x, *d_s;
    cudaMalloc(&d_x, sizeof(float) * size);
    cudaMalloc(&d_s, sizeof(float) * size);

    // Copiar os dados dos dois arrays para dentro da gpu
    cudaMemcpy(d_x, x, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s, sizeof(float) * size, cudaMemcpyHostToDevice);

    // Configuração dos blocos
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Execução do kernel
    add_source_kernel<<<numBlocks, threadsPerBlock>>>(size, d_x, d_s, dt);

    // Copiar os resultados do kernel para o cpu
    cudaMemcpy(x, d_x, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // Limpar a memoria da gpu
    cudaFree(d_x);
    cudaFree(d_s);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------

__global__ void set_bnd_kernel(
    int M, int N, int O, int b, float *x) {

    // Índices 3D globais
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Trabalhar com as bordas no eixo X
    if (i == 0) { // Face esquerda
        x[IX(0, j, k)] = (b == 1) ? -x[IX(1, j, k)] : x[IX(1, j, k)];
    }
    if (i == M + 1) { // Face direita
        x[IX(M + 1, j, k)] = (b == 1) ? -x[IX(M, j, k)] : x[IX(M, j, k)];
    }

    // Trabalhar com as bordas no eixo Y
    if (j == 0) { // Face de baixo
        x[IX(i, 0, k)] = (b == 2) ? -x[IX(i, 1, k)] : x[IX(i, 1, k)];
    }
    if (j == N + 1) { // Face de cima
        x[IX(i, N + 1, k)] = (b == 2) ? -x[IX(i, N, k)] : x[IX(i, N, k)];
    }

    // Trabalhar com as bordas no eixo Z
    if (k == 0) { // Face frontal
        x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
    }
    if (k == O + 1) { // Face traseira
        x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    }

    // Condições nos cantos (processadas apenas pelo thread (0,0,0))
    if (i == 0 && j == 0 && k == 0) {
        x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
    }
}

void set_bnd(int M, int N, int O, int b, float *x) {
    int size = (M + 2) * (N + 2) * (O + 2); // Inclui as bordas

    // Alocar memória na GPU
    float *d_x;
    cudaMalloc(&d_x, sizeof(float) * size);

    // Copiar os dados para a GPU
    cudaMemcpy(d_x, x, sizeof(float) * size, cudaMemcpyHostToDevice);

    // Configuração de threads e blocos
    dim3 threadsPerBlock(8, 8, 8); // 512 threads por bloco
    dim3 numBlocks((M + 7) / 8, (N + 7) / 8, (O + 7) / 8);

    // Executar kernel
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x);

    // Copiar os resultados de volta para a CPU
    cudaMemcpy(x, d_x, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // Liberar memória na GPU
    cudaFree(d_x);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------

__global__ void lin_solve_kernel(float *x, float *x0, float a, float c, int M, int N, int O, int parity, float *max_change){
    // Índices globais (1..M, 1..N, 1..O)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    // Índice local do thread para redução
    int local_id = threadIdx.z * blockDim.y * blockDim.x 
                 + threadIdx.y * blockDim.x
                 + threadIdx.x;

    // Memória compartilhada dinâmica, para fazer redução local
    extern __shared__ float sdata[];
    sdata[local_id] = 0.0f;

    // Se estiver dentro do domínio e a paridade bater, faz a atualização
    if (i <= M && j <= N && k <= O && ((i + j + k) % 2 == parity))
    {
        int idx = IX(i, j, k);

        float old_x = x[idx];
        x[idx] = (x0[idx] +
                  a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                       x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                       x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * c;

        float change = fabsf(x[idx] - old_x);
        // Armazena localmente para reduzir
        sdata[local_id] = change;
    }

    __syncthreads();

    // Redução em log2(numThreads) passos
    int stride = blockDim.x * blockDim.y * blockDim.z / 2;
    while (stride > 0)
    {
        if (local_id < stride)
        {
            sdata[local_id] = fmaxf(sdata[local_id], sdata[local_id + stride]);
        }
        __syncthreads();
        stride /= 2;
    }

    // Thread 0 do bloco escreve o max local em max_change (global)
    if (local_id == 0)
    {
        atomicMax((int*) max_change, __float_as_int(sdata[0]));
    }
}

// ----------------------------------------------------------------------------
// Solver que chama o kernel otimizado
// ----------------------------------------------------------------------------

void lin_solve(int M, int N, int O, int b, float * __restrict__ x, float * __restrict__ x0, float a, float c){
    const float tol = 1e-7f;
    const float inv_c = 1.0f / c;

    // Configuração de threads e blocos
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks( (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                    (O + threadsPerBlock.z - 1) / threadsPerBlock.z );

    // Alocar vetores no device
    float* d_x;
    float* d_x0;
    float* d_max_change;
    cudaMalloc(&d_x,  sizeof(float) * (M+2)*(N+2)*(O+2));
    cudaMalloc(&d_x0, sizeof(float) * (M+2)*(N+2)*(O+2));
    cudaMalloc(&d_max_change, sizeof(float));

    // Copiar dados para o device
    cudaMemcpy(d_x,  x,  sizeof(float) * (M+2)*(N+2)*(O+2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x0, x0, sizeof(float) * (M+2)*(N+2)*(O+2), cudaMemcpyHostToDevice);

    int l = 0;
    float max_change;
    do {
        max_change = 0.0f;
        cudaMemcpy(d_max_change, &max_change, sizeof(float), cudaMemcpyHostToDevice);

        // Lançar kernel para paridade 0
        lin_solve_kernel<<<
            numBlocks, 
            threadsPerBlock,
            threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(float)
        >>>(d_x, d_x0, a, inv_c, M, N, O, 0, d_max_change);

        // Lançar kernel para paridade 1
        lin_solve_kernel<<<
            numBlocks, 
            threadsPerBlock,
            threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(float)
        >>>(d_x, d_x0, a, inv_c, M, N, O, 1, d_max_change);

        // Uma só sincronização no final
        cudaDeviceSynchronize();

        // Copiar de volta a mudança máxima
        cudaMemcpy(&max_change, d_max_change, sizeof(float), cudaMemcpyDeviceToHost);

        l++;

    } while (max_change > tol && l < 20);

    // Copiar resultados de volta
    cudaMemcpy(x, d_x, sizeof(float) * (M+2)*(N+2)*(O+2), cudaMemcpyDeviceToHost);

    // Liberar memória
    cudaFree(d_x);
    cudaFree(d_x0);
    cudaFree(d_max_change);

    // Impor condições de fronteira (no final, pois b pode ser != 0)
    set_bnd(M, N, O, b, x);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------

void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------

__global__ void advect_kernel_shared(
    int M, int N, int O, int b, float *d, const float *d0,
    const float *u, const float *v, const float *w, float dt) {

    // Shared memory para os dados locais
    __shared__ float s_d0[8][8][8]; // Ajustar para o tamanho do bloco

    // Índices 3D globais
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    // Índices locais no bloco
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int lz = threadIdx.z;

    // Calcular constantes
    float dtX = dt * M;
    float dtY = dt * N;
    float dtZ = dt * O;

    // Verificar se estamos dentro dos limites
    if (i <= M && j <= N && k <= O) {
        int idx = IX(i, j, k);

        // Carregar dados para a memória shared
        s_d0[lx][ly][lz] = d0[idx];
        __syncthreads(); // Sincronizar antes de usar os dados carregados

        // Calcular novas posições
        float x = i - dtX * u[idx];
        float y = j - dtY * v[idx];
        float z = k - dtZ * w[idx];

        // Limitar as posições
        x = fmaxf(0.5f, fminf(M + 0.5f, x));
        y = fmaxf(0.5f, fminf(N + 0.5f, y));
        z = fmaxf(0.5f, fminf(O + 0.5f, z));

        // Interpolar valores
        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d[idx] =
            s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                  t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
            s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                  t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
    }
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2); // Tamanho total

    // Alocar memória na GPU
    float *d_d, *d_d0, *d_u, *d_v, *d_w;
    cudaMalloc(&d_d, sizeof(float) * size);
    cudaMalloc(&d_d0, sizeof(float) * size);
    cudaMalloc(&d_u, sizeof(float) * size);
    cudaMalloc(&d_v, sizeof(float) * size);
    cudaMalloc(&d_w, sizeof(float) * size);

    // Copiar dados para a GPU
    cudaMemcpy(d_d, d, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d0, d0, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, sizeof(float) * size, cudaMemcpyHostToDevice);

    // Configuração de threads e blocos
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + 7) / 8, (N + 7) / 8, (O + 7) / 8);

    // Executar kernel
    advect_kernel_shared<<<numBlocks, threadsPerBlock>>>(
        M, N, O, b, d_d, d_d0, d_u, d_v, d_w, dt);

    // Copiar resultado de volta para a CPU
    cudaMemcpy(d, d_d, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // Liberar memória
    cudaFree(d_d);
    cudaFree(d_d0);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);

    // Aplicar condições de contorno na CPU (ou implementar em CUDA depois)
    set_bnd(M, N, O, b, d);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------

void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    const float h = -0.5f / MAX(M, MAX(N, O));

    #pragma omp parallel for collapse(3)
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                int idx = IX(i, j, k);
                float du = u[idx + 1] - u[idx - 1];
                float dv = v[idx + (M + 2)] - v[idx - (M + 2)];
                float dw = w[idx + (M + 2) * (N + 2)] - w[idx - (M + 2) * (N + 2)];
                div[idx] = h * (du + dv + dw);
                p[idx] = 0.0f;
            }
        }
    }
    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    lin_solve(M, N, O, 0, p, div, 1, 6);

    #pragma omp parallel for collapse(3)
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                int idx = IX(i, j, k);
                u[idx] -= 0.5f * (p[idx + 1] - p[idx - 1]);
                v[idx] -= 0.5f * (p[idx + (M + 2)] - p[idx - (M + 2)]);
                w[idx] -= 0.5f * (p[idx + (M + 2) * (N + 2)] - p[idx - (M + 2) * (N + 2)]);
            }
        }
    }

    set_bnd(M, N, O, 1, u);
    set_bnd(M, N, O, 2, v);
    set_bnd(M, N, O, 3, w);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
    add_source(M, N, O, x, x0, dt);
    SWAP(x0, x);
    diffuse(M, N, O, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------

void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {
    add_source(M, N, O, u, u0, dt);
    add_source(M, N, O, v, v0, dt);
    add_source(M, N, O, w, w0, dt);

    SWAP(u0, u);
    diffuse(M, N, O, 1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(M, N, O, 2, v, v0, visc, dt);
    SWAP(w0, w);
    diffuse(M, N, O, 3, w, w0, visc, dt);

    project(M, N, O, u, v, w, u0, v0);

    SWAP(u0, u);
    SWAP(v0, v);
    SWAP(w0, w);

    advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
    advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
    advect(M, N, O, 3, w, w0, u0, v0, w0, dt);

    project(M, N, O, u, v, w, u0, v0);
}
