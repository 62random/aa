#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define 			RANDOM_GEN 			0
#define 			ALL_1				1
#define 			ONLY_ALLOC			2
#define				BLOCK_SIZE			64

int SIZE;

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
    if (opt != 2)
        for(int i = 0; i < SIZE; i++)
            for(int j = 0; j < SIZE; j++)
                if(opt == 1)
                    matrix[i*SIZE + j]  = 1.0;
                else
                    matrix[i*SIZE + j]  = (float) sin(i+j);
                    //matrix[i*SIZE + j]  = (float(rand())/float((RAND_MAX)) * a);
    return matrix;
}

// Versões originais

void matrixMultIJK(float * matrix_a, float * matrix_b, float * matrix_c){
    float sum;
    int i, j, k;

    for( i = 0; i < SIZE; i ++)
        for( j = 0; j < SIZE; j++){
            sum = 0;
            for ( k = 0; k < SIZE; k++ )
                sum += matrix_a[i*SIZE + k] * matrix_b[k*SIZE + j] ;
            matrix_c[i*SIZE + j] = sum;
        }
}

void matrixMultIKJ(float * matrix_a, float * matrix_b, float * matrix_c){
    int sum;
    int i, j, k;

    for( i = 0; i < SIZE; i ++) {
        for( k = 0; k < SIZE; k++){
            for ( j = 0; j < SIZE; j++ )
                matrix_c[i*SIZE + j] += matrix_a[i*SIZE + k] * matrix_b[k*SIZE + j] ;
        }
    }
}


void matrixMultJKI(float * matrix_a, float * matrix_b, float * matrix_c){
    int sum;
    int i, j, k;

    for( j = 0; j < SIZE; j++){
        for( k = 0; k < SIZE; k++){
            for ( i = 0; i < SIZE; i++ )
                matrix_c[i*SIZE + j] += matrix_a[i*SIZE + k] * matrix_b[k*SIZE + j] ;
        }
    }
}

// Versões com transposta sem blocking

void matrixMultIJK_transpose(float * matrix_a, float * matrix_b, float * matrix_c){
    float sum;
    int i, j, k;

    transpose(matrix_b);
    for( i = 0; i < SIZE; i ++)
        for( j = 0; j < SIZE; j++){
            sum = 0;
            for ( k = 0; k < SIZE; k++ )
                sum += matrix_a[i*SIZE + k] * matrix_b[j*SIZE + k];
            matrix_c[i*SIZE + j] = sum;
        }
}

void matrixMultIKJ_transpose(float * matrix_a, float * matrix_b, float * matrix_c){
    int i, j, k;
    for( i = 0; i < SIZE; i ++)
        for( k = 0; k < SIZE; k++){
            for ( j = 0; j < SIZE; j++ )
                matrix_c[i*SIZE + j] += matrix_a[i*SIZE + k] * matrix_b[k*SIZE + j];
        }
}


void matrixMultJKI_transpose(float * matrix_a, float * matrix_b, float * matrix_c){
    int i, j, k;
    transpose(matrix_a);
    for( j = 0; j < SIZE; j ++)
        for( k = 0; k < SIZE; k++){
            for ( i = 0; i < SIZE; i++ )
                matrix_c[i*SIZE + j] += matrix_a[k*SIZE + i]  * matrix_b[k*SIZE + j];
        }
}


// Main

int main(int argc, char *argv[]) {
    float *matrix_a, *matrix_b, *matrix_c;
    float *matrix_aa, *matrix_bb, *matrix_cc;
    SIZE = atoi(argv[1]);

    matrix_a = createMatrix(RANDOM_GEN);
    matrix_b = createMatrix(ALL_1);
    matrix_c = createMatrix(ONLY_ALLOC);
    matrix_aa = createMatrix(RANDOM_GEN);
    matrix_bb = createMatrix(ALL_1);
    matrix_cc = createMatrix(ONLY_ALLOC);

    matrixMultIJK(matrix_a, matrix_b, matrix_c);


    matrixMultJKI_transpose(matrix_aa, matrix_bb, matrix_cc);



    for (int i = 0; i < SIZE; i++)
        for(int j = 0; j < SIZE; j++)
            if( matrix_c[i*SIZE + j]  != matrix_cc[i*SIZE + j] ) {
                printf("Erro\n");
                return -1;
            }

    free(matrix_a);
    free(matrix_b);
    free(matrix_c);
    free(matrix_aa);
    free(matrix_bb);
    free(matrix_cc);

    printf("Finalmente caralho\n");
    return 0;
}
