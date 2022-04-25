
__kernel void fixColumnKernel(__global float *matrix, int size, int rowId){

	__global float row[100];

	__global float Aii;

	int colId = get_global_id(0);

	row[colId] = matrix[size*rowId + colId];
	Aii = matrix[size*rowId + rowId];

	row[colId] = row[colId]/Aii;
	matrix[size*rowId + colId] = row[colId];
	


}
