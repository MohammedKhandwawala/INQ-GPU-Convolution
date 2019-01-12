/*
This code take the kernel and unpacks the kernel(each kernel saperatly) and
stores the position in the orignal kernel of the weights and keep the same weights
together. Now using output tiled convoltion , for each output value position
in input with similar weights are added together and multiplied  similar weights 
*/
#include<stdio.h>
#include<cuda.h>
#include<math.h>
#define CUDA_CALL(x) do { cudaError_t err=(x); \
	if(err!=cudaSuccess) { \
	printf("Error %s at %s: %d",cudaGetErrorString(err),__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)  
#define W 32  // Input DIM
#define D 4   // Input and Kernel Depth
#define T 5   // Kernel DIM
#define N 2   // Number of kernels
#define TILE_W 16 //Tile Width
#define n1 3  //range of INQ n1 > n2
#define n2 1  
#define BAND 3  //Number of unique powers
#define STRIDE_LENGTH 1	//stride length	
#define OWS (W- T + 1)  // Output DIM
#define OW (((W - T)/STRIDE_LENGTH) + 1) //output dim

//comparison operator for sorting
int compare(const void * a, const void * b) 
{ 
    return ( *(int*)a - *(int*)b ); 
} 

//filling the matrix 
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

//fill kernel with weight and its position in the kernel
void fillKernel(int *positions){

int (*p)[4]=(int (*)[4])positions;

for(int i=0;i<N;i++){
	for(int j=0;j<T;j++){
		for(int k=0;k<T;k++){
			for(int l=0;l<D;l++){
				p[i*T*T*D+j*T*D+k*D+l][0]=((i+j+T+D)%n1 + n2);
				p[i*T*T*D+j*T*D+k*D+l][1]=j;
				p[i*T*T*D+j*T*D+k*D+l][2]=k;
				p[i*T*T*D+j*T*D+k*D+l][3]=l;
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
				fprintf(f,"%.4f ", mat[i][j][k]);
			}
			fprintf(f, "\n" );
		}
		fprintf(f,"\n");
	}
	fclose(f);
}

//kernel
__global__ void conv(unsigned char* Dm, float* Do, int* Dp ,int* Dmarkers)
{
	__shared__ int ker[4*T*T*D];	// kernel to shared memory
	__shared__ unsigned char tile[(TILE_W)*(TILE_W)*D];

	int tx=blockDim.x*blockIdx.x+threadIdx.x;
	int ty=blockDim.y*blockIdx.y+threadIdx.y;
	int bz=blockIdx.z;
	int zk=bz*T*T*D;
	int ym,xm;
	int d=0;
	int i=0;
	int j=0;
	int mark[2];
		//mark contain points that saperate unique weights
		mark[0] = Dmarkers[bz*BAND + 1];
		mark[1] = Dmarkers[bz*BAND + 2];
		__syncthreads();
		//copying kernel and position in the shared memory
		for(d=0;d<D;d++)
		{
			if(threadIdx.x<T&&threadIdx.y<T){
				ker[threadIdx.y*4*T*D+threadIdx.x*4*D+4*d]  =Dp[4*zk + threadIdx.y*4*T*D + threadIdx.x*4*D+4*d + 0];
				ker[threadIdx.y*4*T*D+threadIdx.x*4*D+4*d+1]=Dp[4*zk + threadIdx.y*4*T*D + threadIdx.x*4*D+4*d + 1];
				ker[threadIdx.y*4*T*D+threadIdx.x*4*D+4*d+2]=Dp[4*zk + threadIdx.y*4*T*D + threadIdx.x*4*D+4*d + 2];
				ker[threadIdx.y*4*T*D+threadIdx.x*4*D+4*d+3]=Dp[4*zk + threadIdx.y*4*T*D + threadIdx.x*4*D+4*d + 3];
			}
		}

		__syncthreads();
		//copying tile
		for(d=0;d<D;d++)
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
	//output stationary multipllication code
	if( ty%STRIDE_LENGTH == 0 && tx%STRIDE_LENGTH == 0 )
	{								
		int sum[BAND];					
		for(i=0;i<BAND;i++){			
			sum[i]=0;					
		}									
		for(i=0;i<T;i++)
		{				
			int yk1=i*4*T*D;			
			for(j=0;j<T;j++)		
			{						
				int xk1=j*4*D;		
				for(d=0;d<D;d++){
					if (yk1+xk1+d*4 < mark[0]) {//makers contains the points where weights change
						sum[0] += tile[(ker[yk1+xk1+d*4+1] + threadIdx.y)*(TILE_W)*D + (ker[yk1+xk1+d*4+2] + threadIdx.x)*D + ker[yk1+xk1+d*4+3]];//<<ker[yk1+xk1+d*4+0];
					} else if (yk1+xk1+d*4 < mark[1]) {
						sum[1] += tile[(ker[yk1+xk1+d*4+1] + threadIdx.y)*(TILE_W)*D + (ker[yk1+xk1+d*4+2] + threadIdx.x)*D + ker[yk1+xk1+d*4+3]];//<<ker[yk1+xk1+d*4+0];
					} else {    		
						sum[2] += tile[(ker[yk1+xk1+d*4+1] + threadIdx.y)*(TILE_W)*D + (ker[yk1+xk1+d*4+2] + threadIdx.x)*D + ker[yk1+xk1+d*4+3]];//<<ker[yk1+xk1+d*4+0];
					}				
				}					
			}						
		}								
		if(tx<OWS&&ty<OWS){ 		
			Do[bz*OW*OW+(ty/STRIDE_LENGTH)*OW+(tx/STRIDE_LENGTH)] = sum[0]*2+sum[1]*4+sum[2]*8;
		}
	}
}

int main()
{
	//allocating memory on the host for kernel matrix and markers
	int *positions = (int*)malloc(sizeof(int)*4*T*T*D*N);
	unsigned char *matrix = (unsigned char*)malloc(sizeof(unsigned char)*W*W*D);
	float *output = (float *)malloc(sizeof(float)*N*OW*OW);
	int *markers = (int*)malloc(sizeof(int)*3*N);

	//filling kernel and the matrix
	fillMatrix(matrix);
	fillKernel(positions);

	//sorting each kernel (N kernels saperately, so that same weights occur together for each kernel)
	for(int i = 0; i < N ; i++){
		qsort(positions + T*T*D*4*i ,T*T*D , 4*sizeof(int) , compare);
	}

	//finding markers
	for(int i = 0 ; i < N ; i++){
		int mark = 1;
		markers[i*BAND + 0] = 0;
		for(int j = 1; j < T*T*D ; j++){
			if(positions[4*i*T*T*D + j*4] > positions[4*i*T*T*D + (j-1)*4]){
				markers[i*BAND + mark] = j*4;
				mark++;
			}
		}
	}

	//allocating memory on the GPU for kernel (+ positions) , marker and matrix 
	int *Dmarkers;cudaMalloc(&Dmarkers,sizeof(int)*BAND*N);
	int *Dpositions;cudaMalloc(&Dpositions,sizeof(int)*4*N*T*T*D);
	unsigned char *Dmatrix;cudaMalloc(&Dmatrix,sizeof(unsigned char)*W*W*D);
	float *Doutput;cudaMalloc(&Doutput,sizeof(float)*N*OW*OW);
	int blockdimx=(TILE_W - T + 1);
	int blockdimy=(TILE_W - T + 1);
	int griddimz = N;
	int griddimy = (OWS+blockdimx-1)/blockdimx;
	int griddimx = (OWS+blockdimy-1)/blockdimy;
	
	dim3 blocks(griddimx, griddimy, griddimz);
	dim3 thrds_per_block(blockdimx, blockdimy);

	//copying matrix kernel and markers to GPU
	cudaMemcpy(Dmarkers, markers, sizeof(int)*BAND*N,cudaMemcpyHostToDevice);
	cudaMemcpy(Dmatrix, matrix, sizeof(unsigned char)*W*W*D,cudaMemcpyHostToDevice);
	cudaMemcpy(Dpositions, positions, sizeof(int)*4*T*T*D*N,cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start,0);

	conv<<<blocks,thrds_per_block>>>(Dmatrix, Doutput, Dpositions,Dmarkers);
	
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