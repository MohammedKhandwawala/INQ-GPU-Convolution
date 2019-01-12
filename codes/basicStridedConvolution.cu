/*
This code is the normal strided input stationary convolution code to 
benchmark stridedConvolution.cu

HENCE NO WEIGHT SHARING PROPERTY IS USED
*/

#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include<float.h>						
#define CUDA_CALL(x) do { cudaError_t err=(x); \
	if(err!=cudaSuccess) { \
	printf("Error %s at %s: %d",cudaGetErrorString(err),__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)  

#define W 9 // Input DIM
#define D 4   // Input and Kernel Depth
#define T 3	  // Kernel DIM
#define N 128 // Number of kernels
#define TILE_W 4 //input tile width
#define n1 3 //Range for weights(log2) from INQ
#define n2 1 //where n1 > n2
#define BAND 3 // range for weights
#define STRIDE_LENGTH 1 //STRIDE_LENGTH
#define OWS (W- T + 1) // Output DIM
#define OW (((W - T)/STRIDE_LENGTH) + 1) //output width

__global__ void cudaConvolve(float* output,  float*  kernel, unsigned char *matrix){
/*
one block loads its required tile from the matrix collaboritively
and calculates the values for the number of kernels equalling to blockdim.x
*/		
	__shared__ float shmatrix[TILE_W+T-1][TILE_W+T-1][D];
	__shared__ float shkernel[D][T][T][D];	

	float ds=0.0;
	
	long i=0,j=0,k=0;
	
	long ty = threadIdx.y;
	long tx = threadIdx.x;
	long tz = threadIdx.z;
	
	long z = blockIdx.z*TILE_W+tz;
	long y = blockIdx.y*TILE_W+ty;
	long x = blockIdx.x*blockDim.x + tx;

	//Loading kernel in the shared memory
	if (ty<T && tz<T){
		for(k=0;k<D;++k){
			shkernel[k][tz][ty][tx] = kernel[(x-tx+k)*D*T*T + tz*D*T+ ty*D+ tx];		}	
	}
	__syncthreads();	

	//Loadind matrix tile in the shared memory.
	if ( z>=0 && z <W && y>=0 && y <W){
		shmatrix[tz][ty][tx] = matrix[z*D*W + y* D+ tx];
	}
	else
		shmatrix[tz][ty][tx] = 0.0f;

	__syncthreads();
	
	//Input Stationary Convolution
	if(y%STRIDE_LENGTH == 0 && z%STRIDE_LENGTH == 0){
		if (ty<TILE_W && tz<TILE_W){
			for(k=0;k<D;++k){
				for(i=0;i<T;++i){
					for(j=0;j<T;++j){
						ds += shmatrix[i+tz][ty+j][k]*shkernel[tx][i][j][k];
					}
				}
			}
		}	
		__syncthreads();
		
		if (z<OWS && y<OWS && ty<TILE_W && tz<TILE_W){	
			output[x*OW*OW + (z/STRIDE_LENGTH)*OW + (y/STRIDE_LENGTH)] = ds;
		}
	}

}

void fillMatrix(unsigned char *matrix){

unsigned char (*m)[W][D]=(unsigned char (*)[W][D])matrix;

for(int i=0;i<W;i++){
	for(int j=0;j<W;j++){
		for(int k=0;k<D;k++){
			m[i][j][k]=(i*j+j*k+i*k+i*2+j*3+k*4)%255;
				}
			}
		}
}




void fillKernel(float *kernel){

float (*t)[T][T][D]=(float (*)[T][T][D])kernel;

for(int i=0;i<N;i++){
	for(int j=0;j<T;j++){
		for(int k=0;k<T;k++){
			for(int l=0;l<D;l++){
				if(((i+j+T+D)%n1 + n2) >= 0){
					t[i][j][k][l]=(1<<((i+j+T+D)%n1 + n2))*(pow(-1,i+j));
				}
				if(((i+j+T+D)%n1 + n2) < 0){
					t[i][j][k][l]=(1.0/(1<<(-1*((i+j+T+D)%n1 + n2))))*(pow(-1,i+j));
				}
			}
		}
	}
}
}


void printtofile(float *m){

	const char *fname = "GPU_TEST";
	FILE *f = fopen(fname, "w");

	float (*mat)[OW][OW]=(float (*)[OW][OW])m;		

	for(unsigned i=0; i < N; i++) {
		for(unsigned j=0; j < OW; j++){
			for(unsigned k=0;k<OW;k++){
				fprintf(f,"%4.4f ", mat[i][j][k]);
			}
			fprintf(f, "\n");
		}
		fprintf(f,"\n");
	}
	fclose(f);
}



int main()
{

//allocating memory on the host
	unsigned char *matrix=(unsigned char*)malloc(sizeof(unsigned char)*W*W*D);
	float *kernel=(float*)malloc(sizeof(float)*T*T*D*N);
	float *output=(float*)malloc(sizeof(float)*N*OW*OW);


//filling the matrix and the kernel
	fillMatrix(matrix);
	fillKernel(kernel);

//allocating memory on the GPU
	unsigned char *Dmatrix;cudaMalloc(&Dmatrix,sizeof(unsigned char)*W*W*D);
	float *Dkernel;cudaMalloc(&Dkernel,sizeof(float)*N*T*T*D);
	float *Doutput;cudaMalloc(&Doutput,sizeof(float)*N*OW*OW);

//copying matrix and kernel to the GPU
	cudaMemcpy(Dmatrix, matrix, sizeof(unsigned char)*W*W*D,cudaMemcpyHostToDevice);
	cudaMemcpy(Dkernel, kernel, sizeof(float)*T*T*D*N,cudaMemcpyHostToDevice);

//cuda event to record kernel time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start,0);


	dim3 threads(D,TILE_W+T-1,TILE_W+T-1);
	dim3 blocks(N/D, (W+TILE_W-1)/TILE_W , (W+TILE_W-1)/TILE_W );

	cudaConvolve<<< blocks, threads >>>(Doutput, Dkernel, Dmatrix);
	CUDA_CALL(cudaGetLastError());

	
	cudaDeviceSynchronize();

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f\n",milliseconds);


	cudaMemcpy(output, Doutput, sizeof(float)*N*OW*OW,cudaMemcpyDeviceToHost);

	printtofile(output);

}
