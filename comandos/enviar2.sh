cd ..
echo enviar make
scp 'Makefile' pg55986@s7edu2.di.uminho.pt:
echo ------------ A enviar pasta src para o cluster ------------ 
scp -r 'src' pg55986@s7edu2.di.uminho.pt:
echo ------------ A enviar pasta comandos para o cluster ------------ 
scp -r 'comandos' pg55986@s7edu2.di.uminho.pt: