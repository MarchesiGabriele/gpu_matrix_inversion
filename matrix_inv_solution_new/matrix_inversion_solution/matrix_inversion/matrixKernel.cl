#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void makeAugmentedMatrix(__global double *matrix, __global double *inputMatrix, int matrixOrder){
	/* COL VA DA 0 A 2*MATRIX_ORDER */
	int col = get_global_id(0);

	/* ROW VA DA 1 A MATRIX_ORDER */
	int row = get_global_id(1);

	if(col < matrixOrder){
		matrix[col + (row-1)*matrixOrder*2] = inputMatrix[col + (row-1)*matrixOrder];
	}
	else{
		if((col - matrixOrder) == (row-1)){
			matrix[col + (row-1)*matrixOrder*2] = 1;
		}else{
			matrix[col + (row-1)*matrixOrder*2] = 0;
		}
	}
} 

/* La matrice inversa la vado a mettere dentro il buffer della matrice di input. Per evitare di creare un nuovo buffer*/
__kernel void getInvertedMatrix(__global double *matrix, __global double *inputMatrix, int matrixOrder){
	/* COL VA DA 0 A 2*MATRIX_ORDER */
	int col = get_global_id(0);

	/* ROW VA DA 1 A MATRIX_ORDER */
	int row = get_global_id(1);

	if(col >= matrixOrder){
		inputMatrix[(col-matrixOrder) + (row-1)*matrixOrder] = matrix[col + (row-1)*matrixOrder*2]; 
	}
}
