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

extern "C" void lin_solve_cuda(int M, int N, int O, int b, float* x, float* x0, float a, float c);


void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

void set_bnd(int M, int N, int O, int b, float *x) {
    #pragma omp parallel for
    for (int k = 0; k <= O + 1; k++) {
        for (int j = 0; j <= N + 1; j++) {
            x[IX(0, j, k)] = b == 1 ? -x[IX(1, j, k)] : x[IX(1, j, k)];
            x[IX(M + 1, j, k)] = b == 1 ? -x[IX(M, j, k)] : x[IX(M, j, k)];
        }
    }

    #pragma omp parallel for
    for (int k = 0; k <= O + 1; k++) {
        for (int i = 0; i <= M + 1; i++) {
            x[IX(i, 0, k)] = b == 2 ? -x[IX(i, 1, k)] : x[IX(i, 1, k)];
            x[IX(i, N + 1, k)] = b == 2 ? -x[IX(i, N, k)] : x[IX(i, N, k)];
        }
    }

    #pragma omp parallel for
    for (int j = 0; j <= N + 1; j++) {
        for (int i = 0; i <= M + 1; i++) {
            x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
            x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
        }
    }

    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
    x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
    x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
}   



__global__ void lin_solve_kernel(float *x, float *x0, float a, float c, int M, int N, int O, int parity, float *max_change){

    // Coordenadas globais
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    // Verificar se está dentro dos limites
    if (i > M || j > N || k > O) return;

    // Verifica a paridade
    if ((i + j + k) % 2 != parity) return;

    // Cálculo principal
    int idx = IX(i, j, k);
    float old_x = x[idx];
    x[idx] = (x0[idx] +
              a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                   x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                   x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * c;

    // Calcula a mudança máxima local
    float change = fabsf(x[idx] - old_x);
    atomicMax((int*)max_change, __float_as_int(change)); // Atualiza max_change globa
}


void lin_solve(int M, int N, int O, int b, float * __restrict__ x, float * __restrict__ x0, float a, float c) {
    const float tol = 1e-7f;
    const float inv_c = 1.0f / c;
    int l = 0;

    // Configuração de threads e blocos
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Alocar variáveis no dispositivo
    float* d_x;
    float* d_x0;
    float* d_max_change;
    cudaMalloc(&d_x, sizeof(float) * (M+2)  * (N+2)  * (O+2));
    cudaMalloc(&d_x0, sizeof(float) * (M+2)  * (N+2)  * (O+2)  );
    cudaMalloc(&d_max_change, sizeof(float));

    // Copiar dados para o dispositivo
    cudaMemcpy(d_x, x, sizeof(float) * (M+2)  * (N+2) * (O+2) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_x0, x0, sizeof(float) * (M+2)  * (N+2)  * (O+2) , cudaMemcpyHostToDevice);

    float max_change;
    do {
        max_change = 0.0f;
        cudaMemcpy(d_max_change, &max_change, sizeof(float), cudaMemcpyHostToDevice);

        // Executar kernel para paridade 0
        lin_solve_kernel<<<numBlocks, threadsPerBlock>>>(d_x, d_x0, a, inv_c, M, N, O, 0, d_max_change);
        cudaDeviceSynchronize();

        // Executar kernel para paridade 1
        lin_solve_kernel<<<numBlocks, threadsPerBlock>>>(d_x, d_x0, a, inv_c, M, N, O, 1, d_max_change);
        cudaDeviceSynchronize();

        // Copiar de volta a mudança máxima
        cudaMemcpy(&max_change, d_max_change, sizeof(float), cudaMemcpyDeviceToHost);

    } while (max_change > tol && l < 20);

    // Copiar resultados de volta para o host
    cudaMemcpy(x, d_x, sizeof(float) * (M+2)  * (N+2)  * (O+2) , cudaMemcpyDeviceToHost);

    // Liberar memória do dispositivo
    cudaFree(d_x);
    cudaFree(d_x0);
    cudaFree(d_max_change);
}


void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    float dtX = dt * M;
    float dtY = dt * N;
    float dtZ = dt * O;

    #pragma omp parallel for collapse(3)
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                int idx = IX(i, j, k);
                float u_val = u[idx];
                float v_val = v[idx];
                float w_val = w[idx];
                float x = i - dtX * u_val;
                float y = j - dtY * v_val;
                float z = k - dtZ * w_val;

                if (x < 0.5f) x = 0.5f;
                if (x > M + 0.5f) x = M + 0.5f;
                if (y < 0.5f) y = 0.5f;
                if (y > N + 0.5f) y = N + 0.5f;
                if (z < 0.5f) z = 0.5f;
                if (z > O + 0.5f) z = O + 0.5f;

                int i0 = (int)x, i1 = i0 + 1;
                int j0 = (int)y, j1 = j0 + 1;
                int k0 = (int)z, k1 = k0 + 1;

                float s1 = x - i0, s0 = 1 - s1;
                float t1 = y - j0, t0 = 1 - t1;
                float u1 = z - k0, u0 = 1 - u1;

                d[IX(i, j, k)] =
                    s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                          t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
                    s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                          t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
            }
        }
    }
    set_bnd(M, N, O, b, d);
}

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

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
    add_source(M, N, O, x, x0, dt);
    SWAP(x0, x);
    diffuse(M, N, O, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(M, N, O, 0, x, x0, u, v, w, dt);
}

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
