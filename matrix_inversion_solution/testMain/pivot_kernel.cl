
/* 

Operazione di pivoting eseguita solo su elementi della diagonale Aii == 0, per evitare divisione per zero durante fix_row_kernel. 

<rowId> è la riga in considerazione per cui controllo l elemento sulla diagonale e eseguo somme delle altre rige nel caso l elemento sia zero 

*/


__kernel void pivotElementsKernel(__global float *matrix, int size, int rowId){

	__local float selectedRow[100];

	__local float Aii;

	/* itero colonne per ciascuna riga*/
	int col = get_global_id(0);

	/* itero righe */
	int row = get_global_id(1);

	Aii = matrix[size*rowId + rowId];

	if(Aii == 0){
		/* riempio la riga corrispondente al mio rowId */
		selectedRow[col] = matrix[size*rowId + col];

		for(int i = 0; i<size; i++){
			/* evito di sommare la stessa riga a se stessa */
			if(rowId != row){
				selectedRow[col] = selectedRow[col] + matrix[size*row + col];
			}
		}
		matrix[size*rowId + col] = selectedRow[col];
	}
}





























