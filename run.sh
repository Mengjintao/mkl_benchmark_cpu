M=$1
K=$2
N=$3

echo $M $K $N

echo "Run serial..."
OMP_NUM_THREADS=1 ./sgemm_host.out $M $K $N
echo "Run with 2 threads..."
OMP_NUM_THREADS=2 ./sgemm_host.out $M $K $N
echo "Run with 4 threads..."
OMP_NUM_THREADS=4 ./sgemm_host.out $M $K $N
echo "Run with 8 threads..."
OMP_NUM_THREADS=8 ./sgemm_host.out $M $K $N
echo "Run with 16 threads..."
OMP_NUM_THREADS=16 ./sgemm_host.out $M $K $N
echo "Run with 28 threads..."
OMP_NUM_THREADS=28 ./sgemm_host.out $M $K $N
