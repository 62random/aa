module load gcc/4.9.0
module load papi/5.5.0

make
echo "impl;1;2;3;4;5;6;7;8"
for tamanho in 2048
do 
    for imp in ijk ikj jki ijk_t ikj_t jki_t
    do 
        echo -n  $imp 
        for i in $(seq 1 8)
        do  
            ./bin/singleThread $imp $tamanho
        done
        echo ""
    done 
done 




make clean
