echo ------------ A importar gcc ------------
module load gcc/11.2.0
echo ------------ A compilar programa ------------ 
make
echo ------------ Programa enviado para execução ------------
echo A executar programa...  
srun --partition=cpar ./fluid_sim
echo ------------ Programa executado! ------------ 
ls