
__kernel void fixColumnKernel(__global float *matrix, int size, int colId){

	/* valori colonna matrice*/
	int i = get_global_id(1);

	/* valori riga matrice*/
	int j = get_global_id(0);

	/* colonna indicata da colId */ 
	__local float col[100];	

	/* elemento della riga corrispondente a colId */
	__local float AColIdj;

	/* riga indicata da i */
	__local float colj[100];

	col[i] = matrix[i*size+ colId];

	/* controllo se elemento è diverso da zero, se lo è già non devo fare nulla*/
	if(col[i] != 0){
		colj[i] = matrix[i*size+j];
		AColIdj = matrix[colId*size + j];

		/* controllo  di non essere sulla diagonale */
		if(i != colId){
			colj[i] = colj[i] - AColIdj * col[i];
		}
		matrix[i*size + j] = colj[i];
	}

}




