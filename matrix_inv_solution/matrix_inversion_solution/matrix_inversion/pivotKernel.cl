#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void pivotElementsKernel(__global double *matrix, int size, int rowId, __global double2 *pivot){
	size_t globalId = get_global_id(0);
	size_t localId = get_local_id(0);

	__local double localDataOld[256];
	__local double localDataMax[256];
	__local int maxRow;

	maxRow = (int)(pivot[0].y);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 

	localDataOld[localId] = matrix[size*rowId + globalId];
	localDataMax[localId] = matrix[size*maxRow + globalId];
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 

	if(maxRow != rowId){
		matrix[size*rowId + globalId] = localDataMax[localId];
		matrix[size*maxRow + globalId] = localDataOld[localId];
	}
}
