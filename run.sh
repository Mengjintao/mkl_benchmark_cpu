M=$1
K=$2
N=$3

echo $M $K $N

echo "Run serial..."
OMP_NUM_THREADS=1 ./sgemm_host.out $M $K $N
echo "Run with 16 threads..."
OMP_NUM_THREADS=16 ./sgemm_host.out $M $K $N
