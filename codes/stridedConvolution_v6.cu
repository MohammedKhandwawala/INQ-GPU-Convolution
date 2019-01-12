//slight modification to the version_3
//Used loop unrolling to see inside the for loop computing the output element
//This code only works for Band of 3 as it contains hardcoded values.

#include<stdio.h>
#include<cuda.h>
#include<math.h>
#define CUDA_CALL(x) do { cudaError_t err=(x); \
	if(err!=cudaSuccess) { \
	printf("Error %s at %s: %d",cudaGetErrorString(err),__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)  
#define W 64 // Input DIM
#define D 3  // Input and Kernel Depth
#define T 7  // Kernel DIM
#define N 128 // Number of kernels
#define TILE_W 16 //output tile size
#define n1 3 //range for weights in INQ
#define n2 1 //n1 > n2
#define BAND 3
#define STRIDE_LENGTH 1 //stride length		
#define OWS (W- T + 1) // Output DIM
#define OW (((W - T)/STRIDE_LENGTH) + 1)

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

void fillKernel(int *kernel){

int (*t)[T][T][D][2]=(int (*)[T][T][D][2])kernel;

for(int i=0;i<N;i++){
	for(int j=0;j<T;j++){
		for(int k=0;k<T;k++){
			for(int l=0;l<D;l++){
				t[i][j][k][l][0]=((i+j+T+D)%n1 + n2);
				t[i][j][k][l][1]=(pow(-1,i+j));
			}
		}
	}
}
}



void printtofile(int *m){

	const char *fname = "GPU_TAST";
	FILE *f = fopen(fname, "w");

	int (*mat)[OW][OW]=(int (*)[OW][OW])m;		

	for(unsigned i=0; i < N; i++) {
		for(unsigned j=0; j < OW; j++){
			for(unsigned k=0;k<OW;k++){
				fprintf(f,"%d ", mat[i][j][k]);
			}
			fprintf(f, "\n" );
		}
		fprintf(f,"\n");
	}
	fclose(f);
}

__global__ void conv(unsigned char* Dm, int* Dk, int* Do)
{
	__shared__ int ker[2*T*T*D];
	__shared__ unsigned char tile[(TILE_W)*(TILE_W)*D];
	int tx=blockDim.x*blockIdx.x+threadIdx.x;
	int ty=blockDim.y*blockIdx.y+threadIdx.y;
	int bz=blockIdx.z;
	int zk=bz*T*T*D;
	int ym,xm;
	    /* each thread computes one elemement in the output matrix 
	       kernel conntains log2 of abs of weights and sign
	    */
		for(int d=0;d<D;d++)
		{
			if(threadIdx.x<T&&threadIdx.y<T){
				ker[threadIdx.y*2*T*D+threadIdx.x*2*D+2*d]=Dk[2*zk+threadIdx.y*2*T*D+threadIdx.x*2*D+2*d];
				ker[threadIdx.y*2*T*D+threadIdx.x*2*D+2*d+1]=Dk[2*zk+threadIdx.y*2*T*D+threadIdx.x*2*D+2*d + 1];
			}
		}
		//__syncthreads();
		for(int d=0;d<D;d++)
		{
			ym=ty*W*D;
			xm=tx*D;
			tile[threadIdx.y*(TILE_W)*D+threadIdx.x*D+d]=Dm[ym+xm+d];
			if((tx+(TILE_W - T + 1))<W&&(threadIdx.x+(TILE_W - T + 1))<(TILE_W))
			{
				ym=ty*W*D;
				xm=(tx+(TILE_W - T + 1))*D;
				tile[threadIdx.y*(TILE_W)*D+(threadIdx.x+(TILE_W - T + 1))*D+d]=Dm[ym+xm+d];
			}
			if((ty+(TILE_W - T + 1))<W&&(threadIdx.y+(TILE_W - T + 1))<(TILE_W))
			{
				ym=(ty+(TILE_W - T + 1))*W*D;
				xm=(tx)*D;
				tile[(threadIdx.y+(TILE_W - T + 1))*(TILE_W)*D+(threadIdx.x)*D+d]=Dm[ym+xm+d];
			}
			if(((ty+(TILE_W - T + 1))<W&&(threadIdx.y+(TILE_W - T + 1))<(TILE_W))&&((tx+(TILE_W - T + 1))<W&&(threadIdx.x+(TILE_W - T + 1))<(TILE_W)))
			{
				ym=(ty+(TILE_W - T + 1))*W*D;
				xm=(tx+(TILE_W - T + 1))*D;
				tile[(threadIdx.y+(TILE_W - T + 1))*(TILE_W)*D+(threadIdx.x+(TILE_W - T + 1))*D+d]=Dm[ym+xm+d];
			}
		}
	__syncthreads();

	if(ty%STRIDE_LENGTH == 0 && tx%STRIDE_LENGTH == 0)
	{
		int psum=0;
		int sum[BAND];

		for(int i=0; i < BAND; i++){
			sum[i] = 0.0;	
		}

		for(int i=0;i<T;i++)
		{
			int yk1=i*2*T*D;
			int ym1=(threadIdx.y+i)*(TILE_W)*D;
			for(int j=0;j<T;j++)
			{
				int xk1=j*2*D;
				int xm1=(threadIdx.x+j)*D;
				for(int d=0;d<D;d++){
					//opened the loop and number of total compare statments reduced(including for loop)
					if(ker[yk1+xk1+2*d+1] > 0){
						if(ker[yk1+xk1+2*d] - n2 == 0)
							sum[0]+=tile[ym1+xm1+d];
						else if(ker[yk1+xk1+2*d] - n2 == 1)
							sum[1]+=tile[ym1+xm1+d];
						else if(ker[yk1+xk1+2*d] - n2 == 2)
							sum[2]+=tile[ym1+xm1+d];
					}
					else{
						if(ker[yk1+xk1+2*d] - n2 == 0)
							sum[0]-=tile[ym1+xm1+d];
						else if(ker[yk1+xk1+2*d] - n2 == 1)
							sum[1]-=tile[ym1+xm1+d];
						else if(ker[yk1+xk1+2*d] - n2 == 2)
							sum[2]-=tile[ym1+xm1+d];
					}
				}
			}
		}	
		for(int i =0;i < BAND; i++){
			if(i+n2>0){
				psum+=sum[i]<<(i + n2);
			}
			else{
				psum+=sum[i]>>((-1)*(i + n2));
			}
		}
		if(tx<OWS&&ty<OWS){ 
			Do[bz*OW*OW+(ty/STRIDE_LENGTH)*OW+(tx/STRIDE_LENGTH)]=psum;
		}
	}
}

int main()
{
	//allocaing memory on the host to store kernel and matrix
	unsigned char *matrix=(unsigned char*)malloc(sizeof(unsigned char)*W*W*D);
	int *kernel=(int*)malloc(sizeof(int)*2*T*T*D*N);
	int *output=(int *)malloc(sizeof(int)*N*OW*OW);

	//filling values in the matrix and kernel
	fillMatrix(matrix);
	fillKernel(kernel);

	//allocating memory on the GPU for kernel and matrix
	unsigned char *Dmatrix;cudaMalloc(&Dmatrix,sizeof(unsigned char)*W*W*D);
	int *Dkernel;cudaMalloc(&Dkernel,sizeof(int)*2*N*T*T*D);
	int *Doutput;cudaMalloc(&Doutput,sizeof(int)*N*OW*OW);
	int blockdimx=(TILE_W - T + 1);
	int blockdimy=(TILE_W - T + 1);
	int griddimz=N;
	int griddimy=(OWS+blockdimx-1)/blockdimx;
	int griddimx=(OWS+blockdimy-1)/blockdimy;
	dim3 blocks(griddimx, griddimy, griddimz);
	dim3 thrds_per_block(blockdimx, blockdimy);

	//copying kernel and matrix to the GPU
	cudaMemcpy(Dmatrix, matrix, sizeof(unsigned char)*W*W*D,cudaMemcpyHostToDevice);
	cudaMemcpy(Dkernel, kernel, sizeof(int)*2*T*T*D*N,cudaMemcpyHostToDevice);

	//Cuda events to time the kernel
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start,0);

	//cuda kernel call
	conv<<<blocks,thrds_per_block>>>(Dmatrix, Dkernel, Doutput);
	CUDA_CALL(cudaGetLastError());
	
	cudaDeviceSynchronize();


	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f\n",milliseconds);

	cudaMemcpy(output, Doutput, sizeof(int)*N*OW*OW,cudaMemcpyDeviceToHost);

	//Use print_matrix_to_file function only 
	printtofile(output);

}
