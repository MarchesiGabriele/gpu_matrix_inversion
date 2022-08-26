#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void fixColumnKernel(__global double *matrix, int size, int colIdx, __global double *output){
	size_t globalId = get_global_id(0);
	size_t globalId2 = get_global_id(1);
	size_t localId = get_local_id(0);
	
	__local double currentRow[256];
	__local double otherRow[256];
	__local double AiIdx;

	currentRow[localId] = matrix[globalId2 * size + globalId];
	otherRow[localId] = matrix[colIdx * size + globalId];
	AiIdx = matrix[globalId2 * size + colIdx];
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(AiIdx != 0 && globalId2 != colIdx){
		currentRow[localId] = currentRow[localId] - (AiIdx * otherRow[localId]);
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	output[globalId2*size + globalId] = currentRow[localId];
}
