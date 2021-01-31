cpu:
	icc mkl_mic_dgemm.c -DCPU -O3 -no-intel-extensions -mkl -qopenmp -o dgemm_host.out
	icc mkl_mic_sgemm.c -DCPU -O3 -no-intel-extensions -mkl -qopenmp -o sgemm_host.out
clean:
	rm *.out

