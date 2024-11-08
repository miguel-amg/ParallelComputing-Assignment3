cd ..
cd src
echo ------------ A importar gcc ------------
module load gcc/11.2.0
echo ------------ Declaradas 8 threads ------------
export OMP_NUM_THREADS=8
echo ------------ A compilar programa ------------ 
make
echo ------------ A enviar programa ------------ 
echo A executar...
srun --partition=cpar --cpus-per-task=8 --exclusive perf stat ./fluid_sim