# Solver original
```
// red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c, old_x, change;
    int l = 0;
    
    do {
        max_c = 0.0f;
        for (int i = 1; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                 for (int k = 1 + (i+j)%2; k <= O; k+=2) {
                    old_x = x[IX(i, j, k)];
                    x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                                      a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                           x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                           x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /c;
                    change = fabs(x[IX(i, j, k)] - old_x);
                    if(change > max_c) max_c = change;
                }
            }
        }
        
        for (int i = 1; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                for (int k = 1 + (i+j+1)%2; k <= O; k+=2) {
                    old_x = x[IX(i, j, k)];
                    x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                                      a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                           x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                           x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /c;
                    change = fabs(x[IX(i, j, k)] - old_x);
                    if(change > max_c) max_c = change;
                }
            }
        }
        set_bnd(M, N, O, b, x);
    } while (max_c > tol && ++l < 20);
}
```
# Versão 1
# Alteração da ordem
```
// Linear solve for implicit methods (diffusion)
// red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c, old_x, change;
    int l = 0;

    do {
        max_c = 0.0f;
        
        // Alteração da ordem dos loops para (k, j, i) <----------------------------------------------- ALTERAÇÃO
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1 + (j + k) % 2; i <= M; i += 2) {
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

        // Alteração da ordem dos loops para (k, j, i) <----------------------------------------------- ALTERAÇÃO
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1 + (j + k + 1) % 2; i <= M; i += 2) {
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

        set_bnd(M, N, O, b, x);
    } while (max_c > tol && ++l < 20);
}
```

# Versão 2 (lin_solve) 
```
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c, old_x, change;
    int l = 0;

    do {
        max_c = 0.0f;
        
        // Alteração da ordem dos loops para (k, j, i)
        #pragma omp parallel for reduction(max:max_c) private(old_x, change) // <---------------------------------------------- ADICIONADO PRAGMA
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1 + (j + k) % 2; i <= M; i += 2) {
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

        // Segunda parte do Red-Black com a nova ordem de loops (k, j, i)
        #pragma omp parallel for reduction(max:max_c) private(old_x, change) // <---------------------------------------------- ADICIONADO PRAGMA
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1 + (j + k + 1) % 2; i <= M; i += 2) {
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

        set_bnd(M, N, O, b, x);
    } while (max_c > tol && ++l < 20);
}
```

# Versão 3 (set_bnd)
```
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

  // Set boundary on faces
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= N; j++) {
      x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
      x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    }
  }

  for (i = 1; i <= N; i++) {
    for (j = 1; j <= O; j++) {
      x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
      x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
    }
  }
  
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= O; j++) {
      x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
      x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
    }
  }

  // Set corners
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] =
      0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] =
      0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                    x[IX(M + 1, N + 1, 1)]);
}
```

## Depois 
```
// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

  // Set boundary on faces
  #pragma omp parallel for private(i, j) // <---------------------------------------------- ADICIONADO PRAGMA
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= N; j++) {
      x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
      x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    }
  }

  #pragma omp parallel for private(i, j) // <---------------------------------------------- ADICIONADO PRAGMA
  for (i = 1; i <= N; i++) {
    for (j = 1; j <= O; j++) {
      x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
      x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
    }
  }

  #pragma omp parallel for private(i, j) // <---------------------------------------------- ADICIONADO PRAGMA
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= O; j++) {
      x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
      x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
    }
  }

  // Set corners
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] =
      0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] =
      0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                    x[IX(M + 1, N + 1, 1)]);
}
```

# Versão 4 (add_source)
## Antes
```
// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2); 

  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}
```

## Depois
```
// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);  // ------------------> 84+2 * 84+2 * 84+2 = 422 iterações, como é chamada 3 vezes temos 1266 iterações com esta instrução, paralelizar vai dar um ganho pequeno
 
  #pragma omp parallel for // <---------------------------------------------- ADICIONADO PRAGMA
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}
```

# Versão 5 (advect)
## Antes
```
// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
  float dtX = dt * M;
  float dtY = dt * N;
  float dtZ = dt * O;

  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        int idx = IX(i, j, k);
        float u_val = u[idx];
        float v_val = v[idx];
        float w_val = w[idx];

        // Cálculo das posições retroativas
        float x = i - dtX * u_val;
        float y = j - dtY * v_val;
        float z = k - dtZ * w_val;

        // Clamp to grid boundaries
        if (x < 0.5f)
          x = 0.5f;
        if (x > M + 0.5f)
          x = M + 0.5f;
        if (y < 0.5f)
          y = 0.5f;
        if (y > N + 0.5f)
          y = N + 0.5f;
        if (z < 0.5f)
          z = 0.5f;
        if (z > O + 0.5f)
          z = O + 0.5f;

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
```

## Depois
```
// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
  float dtX = dt * M;
  float dtY = dt * N;
  float dtZ = dt * O;

  #pragma omp parallel for shared(d, d0, u, v, w) // <---------------------------------------------- ADICIONADO PRAGMA
  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        int idx = IX(i, j, k);
        float u_val = u[idx];
        float v_val = v[idx];
        float w_val = w[idx];

        // Cálculo das posições retroativas
        float x = i - dtX * u_val;
        float y = j - dtY * v_val;
        float z = k - dtZ * w_val;

        // Clamp to grid boundaries
        if (x < 0.5f)
          x = 0.5f;
        if (x > M + 0.5f)
          x = M + 0.5f;
        if (y < 0.5f)
          y = 0.5f;
        if (y > N + 0.5f)
          y = N + 0.5f;
        if (z < 0.5f)
          z = 0.5f;
        if (z > O + 0.5f)
          z = O + 0.5f;

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
```

# Versão 6 (project)
## Antes
```
// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
  const float max= -0.5f/MAX(M, MAX(N, O));

  for (int k = 1; k <= O; k++) {          
    for (int j = 1; j <= N; j++) {       
      int idx_base = IX(0, j, k);       
      for (int i = 1; i <= M; i++) {     
        int idx = idx_base + i;          

        float du= u[idx + 1] - u[idx - 1];                              
        float dv= v[idx + (M + 2)] - v[idx - (M + 2)];                     
        float dw= w[idx + (M + 2) * (N + 2)] - w[idx - (M + 2) * (N + 2)]; 

        div[idx]= max * (du + dv + dw); 
        p[idx] = 0;                     
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  for (int k = 1; k <= O; k++) {     
    for (int j = 1; j <= N; j++) {  
      int idx_base = IX(0, j, k);   
      for (int i = 1; i <= M; i++) { 
        int idx = idx_base + i;      
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
```

## Depois
```
// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
  const float max= -0.5f/MAX(M, MAX(N, O));

  #pragma omp parallel for  // <---------------------------------------------- ADICIONADO PRAGMA
  for (int k = 1; k <= O; k++) {          // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
    for (int j = 1; j <= N; j++) {        // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
      int idx_base = IX(0, j, k);         // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
      for (int i = 1; i <= M; i++) {      // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
        int idx = idx_base + i;           // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI

        float du= u[idx + 1] - u[idx - 1];                                 // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
        float dv= v[idx + (M + 2)] - v[idx - (M + 2)];                     // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
        float dw= w[idx + (M + 2) * (N + 2)] - w[idx - (M + 2) * (N + 2)]; // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI

        div[idx]= max * (du + dv + dw); // NÃO LEVANTA DATA RACE, confirmei atráves de calculos matemáticos
        p[idx] = 0;                     // NÃO LEVANTA DATA RACE, confirmei atráves de calculos matemáticos
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  #pragma omp parallel for  // <---------------------------------------------- ADICIONADO PRAGMA
  for (int k = 1; k <= O; k++) {     // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
    for (int j = 1; j <= N; j++) {   // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
      int idx_base = IX(0, j, k);    // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
      for (int i = 1; i <= M; i++) { // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
        int idx = idx_base + i;      // NÃO LEVANTA DATA RACE PORQUE É DECLARADA AQUI
        u[idx] -= 0.5f * (p[idx + 1] - p[idx - 1]);                                 // NÃO LEVANTA DATA RACE, confirmei atráves de calculos matemáticos
        v[idx] -= 0.5f * (p[idx + (M + 2)] - p[idx - (M + 2)]);                     // NÃO LEVANTA DATA RACE, confirmei atráves de calculos matemáticos
        w[idx] -= 0.5f * (p[idx + (M + 2) * (N + 2)] - p[idx - (M + 2) * (N + 2)]); // NÃO LEVANTA DATA RACE, confirmei atráves de calculos matemáticos
      }
    }
  }

  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}
```