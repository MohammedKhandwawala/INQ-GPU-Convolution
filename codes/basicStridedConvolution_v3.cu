/*	Normal strided convolution output tiled code to benchmark
	stridedConvolution_v3.cu .
*/ 
#include<stdio.h>
#include<cuda.h>
#include<math.h>
#define CUDA_CALL(x) do { cudaError_t err=(x); \
	if(err!=cudaSuccess) { \
	printf("Error %s at %s: %d",cudaGetErrorString(err),__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)  
#define W 32 // Input DIM
#define D 4  // Input and Kernel Depth
#define T 5  // Kernel DIM
#define N 2 // Number of kernels
#define TILE_W 16 //Tile Size
#define n1 3 //range of weights from INQ
#define n2 1 //n1>n2
#define BAND 3 
#define STRIDE_LENGTH 1		
#define OWS (W- T + 1) // Output DIM
#define OW (((W - T)/STRIDE_LENGTH) + 1) //output DIM


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
					t[i][j][k][l]=(1<<((i+j+T+D)%n1 + n2));//*(pow(-1,i+j));
				}
				if(((i+j+T+D)%n1 + n2) < 0){
					t[i][j][k][l]=(1.0/(1<<(-1*((i+j+T+D)%n1 + n2))));//*(pow(-1,i+j));
				}
			}
		}
	}
}
}


void printtofile(float *m){

	const char *fname = "GPU_TAST";
	FILE *f = fopen(fname, "w");

	float (*mat)[OW][OW]=(float (*)[OW][OW])m;		

	for(unsigned i=0; i < N; i++) {
		for(unsigned j=0; j < OW; j++){
			for(unsigned k=0;k<OW;k++){
				fprintf(f,"%4.4f ", mat[i][j][k]);
			}
			fprintf(f, "\n" );
		}
		fprintf(f,"\n");
	}
	fclose(f);
}

__global__ void conv(unsigned char* Dm, float* Dk, float* Do)
{
	/*  
	    each thread computes one output element
		Output Stationary
	*/

	__shared__ float ker[T*T*D];
	__shared__ unsigned char tile[(TILE_W)*(TILE_W)*D];

	int tx=blockDim.x*blockIdx.x+threadIdx.x;
	int ty=blockDim.y*blockIdx.y+threadIdx.y;
	int bz=blockIdx.z;
	int zk=bz*T*T*D;
	int ym,xm;
		//copying kernel to the shared memory
		for(int d=0;d<D;d++)
		{
			if(threadIdx.x<T&&threadIdx.y<T){
				ker[threadIdx.y*T*D+threadIdx.x*D+d]=Dk[zk+threadIdx.y*T*D+threadIdx.x*D+d];			}
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
		float sum=0.0;
		for(int i=0;i<T;i++)
		{
			int yk1=i*T*D;
			int ym1=(threadIdx.y+i)*(TILE_W)*D;
			for(int j=0;j<T;j++)
			{
				int xk1=j*D;
				int xm1=(threadIdx.x+j)*D;
				for(int d=0;d<D;d++){
					sum+=tile[ym1+xm1+d]*ker[yk1+xk1+d];
				}
			}
		}
		if(tx<OWS&&ty<OWS){ 
			Do[bz*OW*OW+(ty/STRIDE_LENGTH)*OW+(tx/STRIDE_LENGTH)]=sum;
		}
	}
}

int main()
{
	//allocating memory for matrix and kernel on the host
	unsigned char *matrix=(unsigned char*)malloc(sizeof(unsigned char)*W*W*D);
	float *kernel=(float*)malloc(sizeof(float)*T*T*D*N);
	float *output=(float *)malloc(sizeof(float)*N*OW*OW);

	//filling matrix and kernel 
	fillMatrix(matrix);
	fillKernel(kernel);

	//allocating memory for the kernel and matrix on the GPU
	unsigned char *Dmatrix;cudaMalloc(&Dmatrix,sizeof(unsigned char)*W*W*D);
	float *Dkernel;cudaMalloc(&Dkernel,sizeof(float)*N*T*T*D);
	float *Doutput;cudaMalloc(&Doutput,sizeof(float)*N*OW*OW);
	int blockdimx=(TILE_W - T + 1);
	int blockdimy=(TILE_W - T + 1);
	int griddimz=N;
	int griddimy=(OWS+blockdimx-1)/blockdimx;
	int griddimx=(OWS+blockdimy-1)/blockdimy;
	dim3 blocks(griddimx, griddimy, griddimz);
	dim3 thrds_per_block(blockdimx, blockdimy);

	//copying matrix and kernel to the GPU
	cudaMemcpy(Dmatrix, matrix, sizeof(unsigned char)*W*W*D,cudaMemcpyHostToDevice);
	cudaMemcpy(Dkernel, kernel, sizeof(float)*T*T*D*N,cudaMemcpyHostToDevice);

	//cuda events to time kernel
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


	cudaMemcpy(output, Doutput, sizeof(float)*N*OW*OW,cudaMemcpyDeviceToHost);

	//Use print_matrix_to_file function only 
	printtofile(output);

}
