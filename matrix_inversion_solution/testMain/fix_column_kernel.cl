
__kernel void fixColumnKernel(__global float *matrix, int size, int colId){

	int i = get_global_id(0);
	int j = get_global_id(1);

	__local float col[100];	
	__local float AColIdj;
	__local float colj[100];

	col[i] = matrix[i*size+ colId];

	AColIdj = matrix[colId*size + j];

	if(i != colId){
		colj[i] = col[i] - AColIdj * col[i];
	}

	matrix[i*size + j] = col[i];
}




