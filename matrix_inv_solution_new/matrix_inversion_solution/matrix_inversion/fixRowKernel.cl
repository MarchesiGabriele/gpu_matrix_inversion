#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void fixRowKernel(__global double *matrix, int size, int rowId, __global double2 *pivot){
	size_t localId = get_local_id(0);
	size_t globalId = get_global_id(0);		

	__local double row[256];
	__local double Aii;

	row[localId] = matrix[size*rowId + globalId];
	Aii = pivot[0].x;
	
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 

	matrix[size*rowId + globalId] = row[localId]/Aii;
}
