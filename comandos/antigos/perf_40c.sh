cd ..
cd src
echo ------------ A importar gcc ------------
module load gcc/11.2.0
echo ------------ Declaradas 40 threads ------------
export OMP_NUM_THREADS=40
echo ------------ A compilar programa ------------ 
make fluid_sim_par
echo ------------ A enviar programa ------------ 
echo A executar...
srun --partition=cpar --ntasks=1 --cpus-per-task=40 --exclusive perf stat ./fluid_sim