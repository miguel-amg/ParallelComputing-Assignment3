cd ..
cd src
echo
echo /////////////////////////////////
echo CUIDADO NAO EXECUTAR NO CLUSTER! 
echo /////////////////////////////////
echo
echo ------------ A compilar programa ------------ 
make
echo ------------ A executar programa ------------ 
OMP_NUM_THREADS=1 ./fluid_sim