#!/bin/bash

# Nome do script SLURM existente
SLURM_SCRIPT="timeFase2Aux.sh"

# Criar diretório para armazenar os arquivos SLURM
SLURM_DIR="./slurm"
mkdir -p $SLURM_DIR

# Submeter o job ao SLURM com diretório de saída especificado
JOB_ID=$(sbatch --parsable --output=$SLURM_DIR/slurm-%j.out $SLURM_SCRIPT)

# Aguardar a execução do job e imprimir o resultado
OUTPUT_FILE="$SLURM_DIR/slurm-${JOB_ID}.out"

echo "Aguardando a execução do job com ID: $JOB_ID ..."

# Esperar até que o arquivo de saída seja gerado
while [ ! -f "$OUTPUT_FILE" ]; do
    sleep 5
done

# Imprimir o resultado
cat $OUTPUT_FILE