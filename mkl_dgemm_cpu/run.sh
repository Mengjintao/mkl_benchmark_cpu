echo "Run serial..."
OMP_NUM_THREADS=1 ./host.out
echo "Run with 16 threads..."
OMP_NUM_THREADS=16 ./host.out
