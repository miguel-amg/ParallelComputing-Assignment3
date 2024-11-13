cd ..
cd src
echo ------------ A importar gcc ------------
module load gcc/11.2.0
echo ------------ Declaradas 4 threads ------------
export OMP_NUM_THREADS=4
echo ------------ A compilar programa ------------ 
make
echo ------------ A enviar programa ------------ 
echo A executar...
srun --partition=cpar --cpus-per-task=4 --ntasks=1  --exclusive ./fluid_sim