/*
Template code for convolution. CS6023, IITM */
#include<stdio.h>
#include<cuda.h>
#include<math.h>

#define W 1024 // Input DIM
#define OW (W-4) // Output DIM
#define D 8   // Input and Kernel Depth
#define T 5  // Kernel DIM
#define N 128 // Number of kernels

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
			t[i][j][k][l]=fmod(-(i+1)*2.1+(j+1)*3.2-(k+1)*4.8+(l+1)*7.1,1.0);
				}
			}
		}
	}
}



void print_matrix_to_file(float *m){

	const char *fname = "assignment4_out";
	FILE *f = fopen(fname, "w");

	float (*mat)[OW][OW]=(float (*)[OW][OW])m;		

	for(unsigned i=0; i < N; i++) {
		for(unsigned j=0; j < OW; j++)
			for(unsigned k=0;k<OW;k++)
				fprintf(f,"%4.4f ", mat[i][j][k]);
		fprintf(f,"\n");
	}
	fclose(f);
}

__global__ void conv(unsigned char* Dm, float* Dk, float* Do)
{
	__shared__ float ker[T*T*D];
	__shared__ unsigned char tile[20*20*D];
	int tx=blockDim.x*blockIdx.x+threadIdx.x;
	int ty=blockDim.y*blockIdx.y+threadIdx.y;
	int n=blockIdx.z;
	int zk=n*T*T*D;
	int ym,xm;
		for(int d=0;d<D;d++)
		{
			if(threadIdx.x<T&&threadIdx.y<T)
				ker[threadIdx.y*T*D+threadIdx.x*D+d]=Dk[zk+threadIdx.y*T*D+threadIdx.x*D+d];
		}
		//__syncthreads();
		for(int d=0;d<D;d++)
		{
			ym=ty*W*D;
			xm=tx*D;
			tile[threadIdx.y*20*D+threadIdx.x*D+d]=Dm[ym+xm+d];
			if((tx+16)<W&&(threadIdx.x+16)<20)
			{
				ym=ty*W*D;
				xm=(tx+16)*D;
				tile[threadIdx.y*20*D+(threadIdx.x+16)*D+d]=Dm[ym+xm+d];
			}
			if((ty+16)<W&&(threadIdx.y+16)<20)
			{
				ym=(ty+16)*W*D;
				xm=(tx)*D;
				tile[(threadIdx.y+16)*20*D+(threadIdx.x)*D+d]=Dm[ym+xm+d];
			}
			if(((ty+16)<W&&(threadIdx.y+16)<20)&&((tx+16)<W&&(threadIdx.x+16)<20))
			{
				ym=(ty+16)*W*D;
				xm=(tx+16)*D;
				tile[(threadIdx.y+16)*20*D+(threadIdx.x+16)*D+d]=Dm[ym+xm+d];
			}
		}
		__syncthreads();
	if(tx<OW&&ty<OW)
	{
		float sum=0.0;
		for(int i=0;i<T;i++)
		{
			int yk1=i*T*D;
			int ym1=(threadIdx.y+i)*20*D;
			for(int j=0;j<T;j++)
			{
				int xk1=j*D;
				int xm1=(threadIdx.x+j)*D;
				for(int d=0;d<D;d++)
					sum+=tile[ym1+xm1+d]*ker[yk1+xk1+d];
			}
		}
	Do[n*OW*OW+ty*OW+tx]=sum;
	}
}

int main()
{

	unsigned char *matrix=(unsigned char*)malloc(sizeof(unsigned char)*W*W*D);
	float *kernel=(float*)malloc(sizeof(float)*T*T*D*N);
	float *output=(float *)malloc(sizeof(float)*N*OW*OW);


	fillMatrix(matrix);
	fillKernel(kernel);


	unsigned char *Dmatrix;cudaMalloc(&Dmatrix,sizeof(unsigned char)*W*W*D);
	float *Dkernel;cudaMalloc(&Dkernel,sizeof(float)*N*T*T*D);
	float *Doutput;cudaMalloc(&Doutput,sizeof(float)*N*OW*OW);
	int blockdimx=16;
	int blockdimy=16;
	int griddimz=N;
	int griddimy=(OW+blockdimx-1)/blockdimx;
	int griddimx=(OW+blockdimy-1)/blockdimy;
	dim3 blocks(griddimx, griddimy, griddimz);
	dim3 thrds_per_block(blockdimx, blockdimy);
	cudaMemcpy(Dmatrix, matrix, sizeof(unsigned char)*W*W*D,cudaMemcpyHostToDevice);
	cudaMemcpy(Dkernel, kernel, sizeof(float)*T*T*D*N,cudaMemcpyHostToDevice);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start,0);

	//Make your cuda kernel call
	conv<<<blocks,thrds_per_block>>>(Dmatrix, Dkernel, Doutput);

	cudaDeviceSynchronize();


	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f\n",milliseconds);


	cudaMemcpy(output, Doutput, sizeof(float)*N*OW*OW,cudaMemcpyDeviceToHost);

	//Use print_matrix_to_file function only 
	print_matrix_to_file(output);

}