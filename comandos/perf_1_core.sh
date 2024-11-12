cd ..
cd src
echo ------------ A importar gcc ------------
module load gcc/11.2.0
echo ------------ A compilar programa ------------ 
make
echo ------------ A enviar programa para execução ------------ 
srun --partition=cpar --ntasks=1  --exclusive perf stat -e instructions,cycles ./fluid_sim