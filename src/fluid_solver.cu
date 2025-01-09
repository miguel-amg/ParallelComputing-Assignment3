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
// Kernel para a função add_source
__global__ void add_source_kernel(int size, float *x, const float *s, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {x[i] += dt * s[i];}
}

// Função add_source
void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2); 

    // Configuração
    int threadsPerBlock = 512;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Execução do kernel
    add_source_kernel<<<numBlocks, threadsPerBlock>>>(size, x, s, dt);

    // Sincronização
    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------------------------------------------------------------------------------
// Kernel para a função set_bnd
__global__ void set_bnd_kernel(int M, int N, int O, int b, float *x) {
    // Índices 3D globais
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Eixo X
    if (i == 0) {x[IX(0, j, k)] = (b == 1) ? -x[IX(1, j, k)] : x[IX(1, j, k)];}
    if (i == M + 1) {x[IX(M + 1, j, k)] = (b == 1) ? -x[IX(M, j, k)] : x[IX(M, j, k)];}

    // Eixo Y
    if (j == 0) {x[IX(i, 0, k)] = (b == 2) ? -x[IX(i, 1, k)] : x[IX(i, 1, k)];}
    if (j == N + 1) {x[IX(i, N + 1, k)] = (b == 2) ? -x[IX(i, N, k)] : x[IX(i, N, k)];}

    // Eixo Z
    if (k == 0) {x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];}
    if (k == O + 1) {x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];}

    // Thread (0,0,0) apenas
    if (i == 0 && j == 0 && k == 0) {x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);}
}

// Função set_bnd
void set_bnd(int M, int N, int O, int b, float *x) {
    // Configuração
    dim3 threadsPerBlock(8, 8, 8); // 512 threads por bloco
    dim3 numBlocks((M + 7) / 8, (N + 7) / 8, (O + 7) / 8);

    // Executar kernel
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, x);

    // Sincronização
    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------------------------------------------------------------------------------
// Kernel para a função lin_solve
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

    // Thread 0 executa
    if (local_id == 0){atomicMax((int*) max_change, __float_as_int(sdata[0]));}
}

// Função lin_solve
void lin_solve(int M, int N, int O, int b, float * __restrict__ x, float * __restrict__ x0, float a, float c){
    const float tol = 1e-7f;
    const float inv_c = 1.0f / c;

    // Configuração de threads e blocos
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks( (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                    (O + threadsPerBlock.z - 1) / threadsPerBlock.z );


    float* d_max_change;
    cudaMalloc(&d_max_change, sizeof(float));

    int l = 0;
    float max_change;
    do {
        max_change = 0.0f;
        cudaMemcpy(d_max_change, &max_change, sizeof(float), cudaMemcpyHostToDevice);

        // Lançar kernel para paridade 0
        lin_solve_kernel<<<numBlocks, threadsPerBlock,threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(float)>>>(x, x0, a, inv_c, M, N, O, 0, d_max_change);

        // Lançar kernel para paridade 1
        lin_solve_kernel<<<numBlocks, threadsPerBlock, threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(float)>>>(x, x0, a, inv_c, M, N, O, 1, d_max_change);

        // Uma só sincronização no final
        cudaDeviceSynchronize();

        // Copiar de volta a mudança máxima
        cudaMemcpy(&max_change, d_max_change, sizeof(float), cudaMemcpyDeviceToHost);

        l++;

    } while (max_change > tol && l < 20);

    // Liberar memória
    cudaFree(d_max_change);

    // Impor condições de fronteira (no final, pois b pode ser != 0)
    set_bnd(M, N, O, b, x);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------
// Função diffuse
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------
// Kernel para a função advect_kernel
__global__ void advect_kernel(int M, int N, int O, int b, float *d, const float *d0, const float *u, const float *v, const float *w, float dt) {
    // Shared memory
    __shared__ float s_d0[8][8][8]; 
    
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
        __syncthreads(); // Sincronizar 

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

// Função advect
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    // Configuração de threads e blocos
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + 7) / 8, (N + 7) / 8, (O + 7) / 8);

    // Executar kernel
    advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d, d0, u, v, w, dt);

    // Aplicar condições de contorno na CPU (ou implementar em CUDA depois)
    set_bnd(M, N, O, b, d);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------
// Kernel #1 para a função project
__global__ void project_kernel_1 (int M, int N, int O, float *u, float *v, float *w, float *p, float *div, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > 1 && i <= M && j > 1 && j <= N && k > 1 && k <= O) { 
        int idx = IX(i, j, k);
        float du = u[idx + 1] - u[idx - 1];
        float dv = v[idx + (M + 2)] - v[idx - (M + 2)];
        float dw = w[idx + (M + 2) * (N + 2)] - w[idx - (M + 2) * (N + 2)];
        div[idx] = h * (du + dv + dw);
        p[idx] = 0.0f;
    }
}

// Kernel #2 para a função project
__global__ void project_kernel_2 (int M, int N, int O, float *u, float *v, float *w, float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {   
        int idx = IX(i, j, k);
        u[idx] -= 0.5f * (p[idx + 1] - p[idx - 1]);
        v[idx] -= 0.5f * (p[idx + (M + 2)] - p[idx - (M + 2)]);
        w[idx] -= 0.5f * (p[idx + (M + 2) * (N + 2)] - p[idx - (M + 2) * (N + 2)]);
    }
}

// Função project
void project (int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    const float h = -0.5f / MAX(M, MAX(N, O));

    // Configuraração
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + 7) / 8, (N + 7) / 8, (O + 7) / 8);

    // Kernel 1
    project_kernel_1<<<numBlocks, threadsPerBlock>>>(M, N, O, u, v, w, p, div, h);
    
    // Sincronização
    cudaDeviceSynchronize();

    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    lin_solve(M, N, O, 0, p, div, 1, 6);

    // Kernel 2
    project_kernel_2<<<numBlocks, threadsPerBlock>>>(M, N, O, u, v, w, p);

    // Sincronização
    cudaDeviceSynchronize();

    set_bnd(M, N, O, 1, u);
    set_bnd(M, N, O, 2, v);
    set_bnd(M, N, O, 3, w);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------
// Função dens_step
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
    add_source(M, N, O, x, x0, dt);
    SWAP(x0, x);
    diffuse(M, N, O, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------
// Função vel_step
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
