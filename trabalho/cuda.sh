module load gcc/4.9.0
module load papi/5.5.0
module load cuda/7.0.28
export CUDA=yes

make
echo "impl;1;2;3"
for tamanho in 32 128 1024 2048
do
    for imp in $(seq 1 2)
    do
        echo -n  $imp
        for i in $(seq 1 3)
        do
            ./bin/cuda $imp $tamanho
        done
        echo ""
    done
done




make clean
