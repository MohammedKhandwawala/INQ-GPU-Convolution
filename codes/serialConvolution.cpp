/*This is the serial code for strided convolution for verifying the output*/

#include<bits/stdc++.h>
#define W 64 // Input DIM
#define D 4  // Input and Kernel Depth
#define T 5  // Kernel DIM
#define N 2 // Number of kernels
#define n1 3
#define n2 1
#define BAND 3
#define STRIDE_LENGTH 1     
#define OWS (W- T + 1) // Output DIM
#define OW (((W - T)/STRIDE_LENGTH) + 1)
using namespace std;


//same matrix as the other codes
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

//same kernel as used for the other codes
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

    const char *fname = "SerialOut";
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

int main(){

    //allocating arrays for matrix and kernel
    unsigned char *matrix=(unsigned char*)malloc(sizeof(unsigned char)*W*W*D);
    float *kernel=(float*)malloc(sizeof(float)*T*T*D*N);
    float *output=(float *)malloc(sizeof(float)*N*OW*OW);

    //filing matrix and kernel
    fillMatrix(matrix);
    fillKernel(kernel);


    int kCenterX = T / 2;
    int kCenterY = T / 2;
    for(int ll =0 ; ll< N; ll++)
    {
        for(int i=0; i < OWS; i+=STRIDE_LENGTH)              // rows
        {
            for(int j=0; j < OWS; j+=STRIDE_LENGTH)          // columns
            {
                for(int h=i; h< T+i; h++)     // kernel rows
                {
                    for(int w=j; w<T+j; w++) // kernel columns
                    {
                        for(int d =0 ;d< D; d++)
                        {    
                            output[ll*OW*OW + (i/STRIDE_LENGTH)*OW +(j/STRIDE_LENGTH)] += matrix[h*W*D + w*D + d] * kernel[ll*D*T*T+(h-i)*D*T+(w-j)*D+d];
                        }
                    }
                }
            }
        }
    }
    printtofile(output);
}
