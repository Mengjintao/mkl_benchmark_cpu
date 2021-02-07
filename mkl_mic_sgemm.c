/*******************************************************************************
!                             INTEL CONFIDENTIAL
!  Copyright(C) 2001-2010 Intel Corporation. All Rights Reserved.
!  The source code contained  or  described herein and all documents related to
!  the source code ("Material") are owned by Intel Corporation or its suppliers
!  or licensors.  Title to the  Material remains with  Intel Corporation or its
!  suppliers and licensors. The Material contains trade secrets and proprietary
!  and  confidential  information of  Intel or its suppliers and licensors. The
!  Material  is  protected  by  worldwide  copyright  and trade secret laws and
!  treaty  provisions. No part of the Material may be used, copied, reproduced,
!  modified, published, uploaded, posted, transmitted, distributed or disclosed
!  in any way without Intel's prior express written permission.
!  No license  under any  patent, copyright, trade secret or other intellectual
!  property right is granted to or conferred upon you by disclosure or delivery
!  of the Materials,  either expressly, by implication, inducement, estoppel or
!  otherwise.  Any  license  under  such  intellectual property  rights must be
!  express and approved by Intel in writing.
!
!*******************************************************************************
!  Content:
!    Automatically Offloaded SGEMM Example Program Text
!******************************************************************************/

/* System headers */
#include <stdio.h>
#include <stdlib.h>
#include <mm_malloc.h>
//#include <malloc.h>
#include <stdint.h>
#include <omp.h>
#include "mkl.h"
/* Timing */
double dsecnd();
/* Matrices */
static float *A, *B, *C, *R;

void PrintMatrix(float *A, int m, int n)
{
	printf("Matrix %dX%d:\n", m,n);
	int i,j;
	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
			printf("%.3f ", A[i*n+j]);
		printf("\n");
	}
}


int main(int argc, char **argv)
{
	int M = atoi(argv[1]);
	int K = atoi(argv[2]);
	int N = atoi(argv[3]);
	int NumA = M*K;
	int NumB = K*N;
	int NumC = M*N;
        int Bytes = sizeof(float);
        printf("M=%d K=%d N=%d\n", M, K, N);

	float alpha = 1.0, beta = 1.0; /* Scaling factors */
	char transa = 'N', transb = 'N'; /* Transposition options */

	int i, j, t; /* Counters */
	double t1, t2, t3, tMic1, tMic2, tMic; /* Timers */

//	matrix_elements = LD * N;
//	matrix_bytes = sizeof(float) * matrix_elements;

	/* Allocate the matrices */
	A = _mm_malloc(NumA*Bytes, 4);
	if (A == NULL) {
		printf("Could not allocate matrix A\n");
		return -1;
	}

	B = _mm_malloc(NumB*Bytes, 4);
	if (B == NULL) {
		printf("Could not allocate matrix B\n");
		return -1;
	}

	C = _mm_malloc(NumC*Bytes, 4);
	R = _mm_malloc(NumC*Bytes, 4);
	if (C == NULL) {
		printf("Could not allocate matrix C\n");
		return -1;
	}

	/* Initialize the matrices */
	for (i = 0; i < NumA; i++) 
		A[i] = 1.0; 
	for (i = 0; i < NumB; i++) 
		B[i] = 1.0; 
	for (i = 0; i < NumC; i++) 
		C[i] = 0.0;
		R[i] = 0.0;

	for(i=0;i<M;i++)	for(j=0;j<N;j++)
		for(t=0;t<K;t++)
			R[i*N+j] += A[i*K+t] * B[t*N+j];

	/* Typical host/CPU call to SGEMM */
	printf("Computing SGEMM on the host...\n");
	t1 = dsecnd();
	printf("Mutiply A %dX%d and B %dX%d ...\n", M, K, K, N);
	for(i=0; i<1; i++)
//		sgemm(&transa, &transb, &M, &N, &K, &alpha, A, &M, B, &K, &beta, C, &M);
		SGEMM("N", "N", &M, &N, &K, &alpha, A, &M, B, &K, &beta, C, &N);

	t2 = dsecnd();

	PrintMatrix(A,M,K);
	PrintMatrix(B,K,N);
	PrintMatrix(C,M,N);
	PrintMatrix(R,M,N);

/*
	for(i=0;i<M;i++)
	{	
		for(j=0;j<N;j++)
//		if(fabs(R[i*N+j] - C[i*N+j]) >1e-6) 
//			printf("%d %d %f %f\n", i,j,R[i*N+j], C[i*N+j]);
*/
	t3 = t2 - t1;
	printf("Total time computing DGEMM on the host: %.2f msecs\n", t3/20*1000);
	printf("GFLOPS of SGEMM on the host: %.2f GFLOPS\n", (2.0*M*N*K/(1024*1024*1024))/(t3/20));

#if 0
    FILE *fp;
    fp=fopen("matrix_c.txt","w");
       for(i=0;i<N;i++){
        for(j=0;j<N;j++){
   	fprintf(fp,"%f\n",C[i*N+j]);
	}
       }
#endif
	/* Free the matrices */
	_mm_free(A); _mm_free(B); _mm_free(C);
	printf("Done\n");
    return 0;
}
