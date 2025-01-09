#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SIZE 168

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;

///////////////////////////////////////////////////////////////////////////////////////
//                              Secção dedicada ao GPU                               //
///////////////////////////////////////////////////////////////////////////////////////
// Arrays a ser utilizados para o GPU 
float *d_u, *d_v, *d_w, *d_u_prev, *d_v_prev, *d_w_prev;
float *d_dens, *d_dens_prev;

// Alocar espaço para todos os arrays no gpu
void gpu_allocate_arrays(int size) {
    cudaMalloc(&d_u, size * sizeof(float));
    cudaMalloc(&d_v, size * sizeof(float));
    cudaMalloc(&d_w, size * sizeof(float));
    cudaMalloc(&d_u_prev, size * sizeof(float));
    cudaMalloc(&d_v_prev, size * sizeof(float));
    cudaMalloc(&d_w_prev, size * sizeof(float));
    cudaMalloc(&d_dens, size * sizeof(float));
    cudaMalloc(&d_dens_prev, size * sizeof(float));
}

// Limpar a gpu
void gpu_free_arrays() {
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_u_prev);
    cudaFree(d_v_prev);
    cudaFree(d_w_prev);
    cudaFree(d_dens);
    cudaFree(d_dens_prev);
}

// Enviar dados do cpu para o gpu
void cpu_to_gpu_arrays() {
    int size = (M + 2) * (N + 2) * (O + 2);
    cudaMemcpy(d_u, u, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_prev, u_prev, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_prev, v_prev, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_prev, w_prev, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dens, dens, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dens_prev, dens_prev, size * sizeof(float), cudaMemcpyHostToDevice);
}

// Enviar dados do gpu para o cpu
void gpu_to_cpu_arrays() {
    int size = (M + 2) * (N + 2) * (O + 2);
    cudaMemcpy(u, d_u, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w, d_w, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_prev, d_u_prev, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_prev, d_v_prev, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_prev, d_w_prev, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dens, d_dens, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dens_prev, d_dens_prev, size * sizeof(float), cudaMemcpyDeviceToHost);
}

// Inicializar os arrays a 0
void gpu_initialize_arrays(int M, int N, int O) {
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

    // Colocar tudo a 0
    cudaMemset(d_u, 0, size);
    cudaMemset(d_v, 0, size);
    cudaMemset(d_w, 0, size);
    cudaMemset(d_u_prev, 0, size);
    cudaMemset(d_v_prev, 0, size);
    cudaMemset(d_w_prev, 0, size);
    cudaMemset(d_dens, 0, size);
    cudaMemset(d_dens_prev, 0, size);

    // Verificar erros 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro ao inicializar arrays GPU: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
///////////////////////////////////////////////////////////////////////////////////////////

// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  u = new float[size];
  v = new float[size];
  w = new float[size];
  u_prev = new float[size];
  v_prev = new float[size];
  w_prev = new float[size];
  dens = new float[size];
  dens_prev = new float[size];

  if (!u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    u[i] = v[i] = w[i] = u_prev[i] = v_prev[i] = w_prev[i] = dens[i] =
        dens_prev[i] = 0.0f;
  }
}

// Free allocated memory
void free_data() {
  delete[] u;
  delete[] v;
  delete[] w;
  delete[] u_prev;
  delete[] v_prev;
  delete[] w_prev;
  delete[] dens;
  delete[] dens_prev;
}

// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event> &events) {
  for (const auto &event : events) {
    int i = M / 2, j = N / 2, k = O / 2;
    int idx = IX(i, j, k); // Índice calculado

    if (event.type == ADD_SOURCE) {
      // Atualizar no CPU primeiro
      dens[idx] = event.density;

      // Copiar para a GPU
      cudaMemcpy(&d_dens[idx], &dens[idx], sizeof(float), cudaMemcpyHostToDevice);
      
    } else if (event.type == APPLY_FORCE) {
      // Atualizar no CPU primeiro
      u[idx] = event.force.x;
      v[idx] = event.force.y;
      w[idx] = event.force.z;

      // Copiar para a GPU
      cudaMemcpy(&d_u[idx], &u[idx], sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(&d_v[idx], &v[idx], sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(&d_w[idx], &w[idx], sizeof(float), cudaMemcpyHostToDevice);
    }
  }
}

// Function to sum the total density
float sum_density() {
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    total_density += dens[i];
  }
  return total_density;
}

// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    // Apply events to the simulation
    apply_events(events);

    // Perform the simulation steps
    vel_step(M, N, O, d_u, d_v, d_w, d_u_prev, d_v_prev, d_w_prev, visc, dt);
    dens_step(M, N, O, d_dens, d_dens_prev, d_u, d_v, d_w, diff, dt);
  }
}

int main() {
  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  // Get the total number of timesteps from the event file
  int timesteps = eventManager.get_total_timesteps();

  // Allocate and clear data
  if (!allocate_data())
    return -1;
  clear_data();

  //////////////////////////////////////////////////////////////////
  // Alocar espaço no gpu
  int size = (M + 2) * (N + 2) * (O + 2);
  gpu_allocate_arrays(size);
  cpu_to_gpu_arrays();
  //////////////////////////////////////////////////////////////////
  // Run simulation with events
  simulate(eventManager, timesteps);
  //////////////////////////////////////////////////////////////////
  // Recolher os resultados
  gpu_to_cpu_arrays();
  //////////////////////////////////////////////////////////////////

  // Print total density at the end of simulation
  float total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  gpu_free_arrays();
  free_data();

  return 0;
}