#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void finalMaxPivotKernel(__global double2 *values){
	size_t globalId = get_global_id(0);		
	size_t size = get_global_size(0);		

	__local double2 vector[1000];
	vector[globalId] = values[globalId];

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(globalId == 0){
		__private double2 max = (double2)(0.0,0.0);
		for(int i = 0; i<size; i++){
			if(fabs(vector[i].x) > fabs(max.x)){
				max.x = vector[i].x;
				max.y = vector[i].y;
			}
		}
		values[0].x = max.x;
		values[0].y = max.y;
	}
}
