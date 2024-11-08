
# Tempo de execução normal do programa

```
------------ A enviar programa para execução ------------
Total density after 100 timesteps: 140876

 Performance counter stats for './fluid_sim':

      805432523028      instructions              #    1.61  insn per cycle
      498734989545      cycles

     152.465668297 seconds time elapsed

     152.450594000 seconds user
       0.026002000 seconds sys
```


# Primeira mudança
```
// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  
  #pragma omp parallel for <--------------------------------------------------------- ADICIONADO
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}
```

# Resultados: 8 CORES
```
Total density after 100 timesteps: 140876

 Performance counter stats for './fluid_sim':

         168678.16 msec task-clock                #    1.029 CPUs utilized
              1477      context-switches          #    0.009 K/sec
                 9      cpu-migrations            #    0.000 K/sec
             36352      page-faults               #    0.216 K/sec
      511088672351      cycles                    #    3.030 GHz
      292028012242      stalled-cycles-frontend   #   57.14% frontend cycles idle
      808527831667      instructions              #    1.58  insn per cycle
                                                  #    0.36  stalled cycles per insn
       10811371441      branches                  #   64.095 M/sec
          93026470      branch-misses             #    0.86% of all branches

     163.846178113 seconds time elapsed

     168.662432000 seconds user
       0.105066000 seconds sys
```
# Segunda mudança
**Antes**
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
```

**Depois**

```
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

  // Set boundary on faces
  #pragma omp parallel { <--------------------------------------------------------- ADICIONADO
    #pragma omp for private(i, j) <--------------------------------------------------------- ADICIONADO
    for (i = 1; i <= M; i++) {
      for (j = 1; j <= N; j++) {
        x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
        x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
      }
    }

    #pragma omp for private(i, j) <--------------------------------------------------------- ADICIONADO
    for (i = 1; i <= N; i++) {
      for (j = 1; j <= O; j++) {
        x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
        x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
      }
    }
    
    #pragma omp for private(i, j) <--------------------------------------------------------- ADICIONADO
    for (i = 1; i <= M; i++) {
      for (j = 1; j <= O; j++) {
        x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
        x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
      }
    }
  }
```


# Resultados
```
 Performance counter stats for './fluid_sim':

         280175.60 msec task-clock                #    1.674 CPUs utilized
             43384      context-switches          #    0.155 K/sec
                77      cpu-migrations            #    0.000 K/sec
             44346      page-faults               #    0.158 K/sec
      841591249114      cycles                    #    3.004 GHz
      518083098353      stalled-cycles-frontend   #   61.56% frontend cycles idle
      902162270733      instructions              #    1.07  insn per cycle
                                                  #    0.57  stalled cycles per insn
       37949135196      branches                  #  135.448 M/sec
         111871002      branch-misses             #    0.29% of all branches

     167.384944129 seconds time elapsed

     279.959497000 seconds user
       1.134816000 seconds sys
```

# Terceira mudança
**Antes:**
```
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

**Depois:**
```
// red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c, old_x, change;
    int l = 0;
    
    do {
        max_c = 0.0f;
        #pragma omp parallel for reduction(max:max_c) private(old_x, change) <--------------------------------------------------------- ADICIONADO
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
        
        #pragma omp parallel for reduction(max:max_c) private(old_x, change) <--------------------------------------------------------- ADICIONADO
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

# Resultados:
Total density after 100 timesteps: 140876

     107.798324983 seconds time elapsed

     625.423479000 seconds user
       0.529093000 seconds sys
