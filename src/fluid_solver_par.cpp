#include "fluid_solver.h"
#include <cmath>
#include <cstring>
#include <omp.h>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x) \
    { float *tmp = x0; x0 = x; x = tmp; }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

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

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c, old_x, change;
    int l = 0;

    do {
        max_c = 0.0f;

        #pragma omp parallel for collapse(3) reduction(max:max_c) private(old_x, change)
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1; i <= M; i++) {
                    if ((i + j + k) % 2 == 0) {
                        int idx = IX(i, j, k);
                        old_x = x[idx];
                        x[idx] = (x0[idx] +
                                  a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                       x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                       x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;

                        change = fabs(x[idx] - old_x);
                        if (change > max_c) max_c = change;
                    }
                }
            }
        }

        #pragma omp parallel for collapse(3) reduction(max:max_c) private(old_x, change)
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1; i <= M; i++) {
                    if ((i + j + k) % 2 != 0) {
                        int idx = IX(i, j, k);
                        old_x = x[idx];
                        x[idx] = (x0[idx] +
                                  a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                       x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                       x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;

                        change = fabs(x[idx] - old_x);
                        if (change > max_c) max_c = change;
                    }
                }
            }
        }

        set_bnd(M, N, O, b, x);
    } while (max_c > tol && ++l < LINEARSOLVERTIMES);
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
