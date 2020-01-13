#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <papi.h>
#include <sys/time.h>
#include <omp.h>

// Variáveis e defines relacionados com a PAPI
#define             NUM_EVENTS          2
int	Events[NUM_EVENTS] = {PAPI_L3_TCM, PAPI_TOT_INS};
int EventSet = PAPI_NULL, retval;
long long int values[NUM_EVENTS];


#define 			RANDOM_GEN 			0
#define 			ALL_1				1
#define 			ONLY_ALLOC			2
#define				BLOCK_SIZE			32

int SIZE;


double clearcache [30000000];

void clearCache (void) {
	for (unsigned i = 0; i < 30000000; ++i)
		clearcache[i] = i;
}

// Medição do tempo
long long unsigned initial_time;
struct timeval begin;
struct timeval end;

void start (void) {
	gettimeofday(&begin, NULL);
}


void stop () {
	gettimeofday(&end, NULL);
	long long duration = (end.tv_sec-begin.tv_sec)*1000000LL + end.tv_usec-begin.tv_usec;
	printf(";%.3f", ((float) duration) / 1000);
}




void blockingTranspose(float * src) {
    int temp;
    for (int ii = 0; ii < SIZE; ii += BLOCK_SIZE)
        for (int jj = 0; jj < SIZE; jj += BLOCK_SIZE)
            for (int i = ii; i < ii + BLOCK_SIZE; i++)
                for (int j = jj + i + 1; j < jj + BLOCK_SIZE; j++)
                {
                    temp = src[i*SIZE + j] ;
                    src[i*SIZE + j]  = src[j*SIZE + i] ;
                    src[j*SIZE + i]  = temp;
                }
}

void transpose (float * src){
    float tmp;
    for(int i = 0; i < SIZE; i ++)
        for(int j = i + 1; j < SIZE; j++)
        {
            tmp = src[i*SIZE + j] ;
            src[i*SIZE + j]  = src[j*SIZE + i] ;
            src[j*SIZE + i]  = tmp;
        }
}



float * createMatrix(int opt){
    float * matrix = (float *) malloc(sizeof(float) * SIZE * SIZE);

    float a = 5.0;
    if(opt != 2) {
        for(int i = 0; i < SIZE; i++)
            for(int j = 0; j < SIZE; j++)
                if(opt == 1)
                    matrix[i*SIZE + j]  = 1.0;
                else
                    //matrix[i*SIZE + j]  = (float) sin(i+j);
                    matrix[i*SIZE + j]  = (float(rand())/float((RAND_MAX)) * a);
	}
	return matrix;
}

// Versões cuda

__global__ void cudaKernel (float *Da, float *Db, float *Dc, intSIZE) {
  int COL = threadIdx.x + blockDim.x * blockIdx.x;
  int ROW = threadIdx.y + blockDim.y * blockIdx.y;
  int k;

  if (ROW>=SIZE || COL>=v){
    float res=0.0f;
    for(k=0; k<SIZE; ++k){
      res += Da[ROW*SIZE + k] * Db[k*SIZE + COL];
    }
    Dc[ROW*SIZE + COL]=res;
  }
}

void no_blocking(float*a, float*b, float*c, intSIZE) {
  float *Da, *Db, *Dc;
  cudaMalloc( (void**) &Da,SIZE *SIZE *sizeof(float) );
  cudaMalloc( (void**) &Db,SIZE *SIZE *sizeof(float) );
  cudaMalloc( (void**) &Dc,SIZE *SIZE *sizeof(float) );

  cudaMemcpy(Da, a,SIZE *SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Db, b,SIZE *SIZE * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid (SIZE/BLOCK_SIZE+(SIZE%BLOCK_SIZE>0),SIZE/BLOCK_SIZE+(SIZE%BLOCK_SIZE>0) );
  cudaKernel<<<dimGrid, dimBlock>>>(Da, Db, Dc,SIZE);

  cudaDeviceSynchronize();
  cudaMemcpy(c, Dc,SIZE *SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(Da);
  cudaFree(Db);
  cudaFree(Dc);
}


__global__ void cudaBlockKernel(float *a, float *b, float *c, intSIZE){
  float CValue = 0;

  int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  for (int k = 0; k < (BLOCK_SIZE +SIZE - 1)/BLOCK_SIZE; k++) {

    if (k*BLOCK_SIZE + threadIdx.x <SIZE && Row <SIZE)
      As[threadIdx.y][threadIdx.x] = a[Row*SIZE + k*BLOCK_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*BLOCK_SIZE + threadIdx.y <SIZE && Col <SIZE)
      Bs[threadIdx.y][threadIdx.x] = b[(k*BLOCK_SIZE + threadIdx.y)*N + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < BLOCK_SIZE; ++n)
      CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }

  if (Row < SIZE && Col < SIZE)
    c[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
      (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

void w_blocking(float*a, float*b, float*c, int N) {
  float *Da, *Db, *Dc;

  size_t size = SIZE * SIZE * sizeof(float);

  cudaMalloc(&Da, size);
  cudaMalloc(&Db, size);

  cudaMemcpy(Da, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(Db, b, size, cudaMemcpyHostToDevice);

  cudaMalloc(&Dc, size);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((N + dimBlock.x -1) / dimBlock.x, (SIZE + dimBlock.y -1 )/ dimBlock.y);

  cudaBlockKernel<<<dimGrid, dimBlock>>>(Da, Db, Dc, SIZE);

  cudaDeviceSynchronize();

  cudaMemcpy(c, Dc, size, cudaMemcpyDeviceToHost);

  cudaFree(Da);
  cudaFree(Db);
  cudaFree(Dc);
}


// Main

int main(int argc, char *argv[]) {
    if(argc < 2) {
        printf("Argumentos incorretos\n");
        return -1;
    }


    //Inicialização da PAPI
	retval = PAPI_library_init(PAPI_VER_CURRENT);
	retval = PAPI_create_eventset(&EventSet);
	retval = PAPI_add_events(EventSet, Events, NUM_EVENTS);


    float *matrix_a, *matrix_b, *matrix_c;
    float *matrix_aa, *matrix_bb, *matrix_cc;
    SIZE = atoi(argv[2]);
    int imp = atoi(argv[1]);

    matrix_a = createMatrix(RANDOM_GEN);
    matrix_b = createMatrix(ALL_1);
    matrix_c = createMatrix(ONLY_ALLOC);
    matrix_aa = createMatrix(RANDOM_GEN);
    matrix_bb = createMatrix(ALL_1);
    matrix_cc = createMatrix(ONLY_ALLOC);


	start();
    if(imp == 1) {
        bo_block(matrix_a, matrix_b, matrix_c);
    }
    else if(imp == 2) {
        w_block(matrix_a, matrix_b, matrix_c);
    }
    else {
        printf("Nenhuma implementação corresponde ao primeiro argumento!!\n");
        return -1;
    }
	stop();

    printf("/%.9f", values[0]/ (float) values[1]);



    free(matrix_a);
    free(matrix_b);
    free(matrix_c);
    free(matrix_aa);
    free(matrix_bb);
    free(matrix_cc);

    return 0;
}
