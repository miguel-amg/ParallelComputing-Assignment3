cd ..
cd src
echo
echo /////////////////////////////////
echo CUIDADO NAO EXECUTAR NO CLUSTER! 
echo /////////////////////////////////
echo
echo ------------ A compilar programa ------------ 
g++ -Wall -fopenmp -pg main.cpp fluid_solver.cpp EventManager.cpp -o fluid_solver
echo ------------ A executar programa ------------ 
./fluid_solver
echo ------------ A gerar relatorio ------------ 
gprof fluid_solver gmon.out > relatorio.txt
echo Relatorio gerado