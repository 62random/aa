#include <stdlib.h>
#include <stdio.h>

#define 			RANDOM_GEN 			0
#define 			ALL_1				1
#define 			ONLY_ALLOC			2
#define				BLOCK_SIZE			64

void blockingTranspose(float ** src, int size) {
    int temp;
    for (int ii = 0; ii < size; ii += BLOCK_SIZE)
        for (int jj = 0; jj < size; jj += BLOCK_SIZE)
            for (int i = ii; i < ii + BLOCK_SIZE; i++)
                for (int j = jj + i + 1; j < jj + BLOCK_SIZE; j++)
                {
                    temp = src[i][j];
                    src[i][j] = src[j][i];
                    src[j][i] = temp;
                }
}

void transpose (float ** src, int size){
    float tmp;
    for(int i = 0; i < size; i ++)
        for(int j = i + 1; j < size; j++)
        {
            tmp = src[i][j];
            src[i][j] = src[j][i];
            src[j][i] = tmp;
        }
}



void createMatrix(float *** matrix, float size, int opt){
    *(matrix) = (float **) malloc(sizeof(float) * size * size);
    float a = 5.0;
    if (opt != 2)
        for(int i = 0; i < size; i ++)
            for(int j = 0; j < size; j++)
                if(opt == 1)
                    (*matrix)[i][j] = 1.0;
                else
                    (*matrix)[i][j] = (float(rand())/float((RAND_MAX)) * a);
}

// Versões originais

void matrixMultIJK(float** matrix_a, float** matrix_b, float** matrix_c, int size){
    int sum;
    int i, j, k;

    for( i = 0; i < size; i ++)
        for( j = 0; j < size; j++){
            sum = 0;
            for ( k = 0; k < size; k++ )
                sum += matrix_a[i][k] * matrix_b[k][j];
            matrix_c[i][j] = sum;
        }
}

void matrixMultIKJ(float** matrix_a, float** matrix_b, float** matrix_c, int size){
    int sum;
    int i, j, k;

    for( i = 0; i < size; i ++)
        for( k = 0; k < size; k++){
            sum = 0;
            for ( j = 0; j < size; j++ )
                sum += matrix_a[i][k] * matrix_b[k][j];
            matrix_c[i][j] = sum;
        }
}


void matrixMultJKI(float** matrix_a, float** matrix_b, float** matrix_c, int size){
    int sum;
    int i, j, k;

    for( j = 0; j < size; j ++)
        for( k = 0; k < size; k++){
            sum = 0;
            for ( i = 0; i < size; i++ )
                sum += matrix_a[i][k] * matrix_b[k][j];
            matrix_c[i][j] = sum;
        }
}

// Versões com transposta sem blocking

void matrixMultIJK_transpose(float** matrix_a, float** matrix_b, float** matrix_c, int size){
    int sum;
    int i, j, k;

    transpose(matrix_b, size);
    for( i = 0; i < size; i ++)
        for( j = 0; j < size; j++){
            sum = 0;
            for ( k = 0; k < size; k++ )
                sum += matrix_a[i][k] * matrix_b[j][k];
            matrix_c[i][j] = sum;
        }
}

void matrixMultIKJ_transpose(float** matrix_a, float** matrix_b, float** matrix_c, int size){
    int sum;
    int i, j, k;
    for( i = 0; i < size; i ++)
        for( k = 0; k < size; k++){
            sum = 0;
            for ( j = 0; j < size; j++ )
                sum += matrix_a[i][k] * matrix_b[k][j];
            matrix_c[i][j] = sum;
        }
}


void matrixMultJKI_transpose(float** matrix_a, float** matrix_b, float** matrix_c, int size){
    int sum;

    int i, j, k;
    transpose(matrix_a, size);
    for( j = 0; j < size; j ++)
        for( k = 0; k < size; k++){
            sum = 0;
            for ( i = 0; i < size; i++ )
                sum += matrix_a[k][i] * matrix_b[k][j];
            matrix_c[i][j] = sum;
        }
}


// Main

int main(int argc, char *argv[]) {
    float **matrix_a, **matrix_b, **matrix_c;
    float **matrix_aa, **matrix_bb, **matrix_cc;
    int size = atoi(argv[1]);
    createMatrix(&matrix_a, size, RANDOM_GEN);
    createMatrix(&matrix_b, size, ALL_1);
    createMatrix(&matrix_c, size, ONLY_ALLOC);

    matrixMultIJK(matrix_a, matrix_b, matrix_c, size);

    createMatrix(&matrix_aa, size, RANDOM_GEN);
    createMatrix(&matrix_bb, size, ALL_1);
    createMatrix(&matrix_cc, size, ONLY_ALLOC);

    for (int i = 0; i < size; i++)
        for(int j = 0; j < 0; j++)
            if( matrix_c[i][j] != matrix_cc[i][j])
                printf("Erro\n");

    free(matrix_a);
    free(matrix_b);
    free(matrix_c);
    free(matrix_aa);
    free(matrix_bb);
    free(matrix_cc);
    return 0;
}
