#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void maxPivotKernel(__global double *matrix, int size, int colId, __global double2 *output){
	size_t localId = get_local_id(0);
	int workGroupId = get_group_id(0);
	size_t localSize = get_local_size(0);
	int globalId = get_global_id(0);

	__private int limiteLoop = 256;
	__local double2 localData2[256];
	__private int lim = 0;

	localData2[localId] = (double2)(matrix[globalId*size + colId], (double)(globalId));		

	barrier(CLK_LOCAL_MEM_FENCE);

	/* Controllo se colId è superiore o meno all'intero workgroup corrente, se è superiore nessun elemento del workgroup corrente può essere il max */
	if(colId <= (workGroupId*256 + 255)){
		if((size/2) < 256){				
			limiteLoop = (int)(size/2);
		}else if(workGroupId == (int)(floor((double)(size/512)))){		/* size == ordine*2 */
			limiteLoop = (int)((size/2)%256);
		}

		/* Controllo se colId è compreso tra gli elementi del workgroup corrente oppure se è sotto, se è sotto non devo tenerne conto durante la ricerca del max */
		if(colId >= (workGroupId*256))
			lim = limiteLoop-(colId%256);
		else
			lim = limiteLoop;
			
		if(lim%2 != 0){
			localData2[limiteLoop] = (double2)(0.0,0.0);
			lim++;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	
		for(int i = lim >> 1; i > 0; i>>=1){
			if(lim < i){
				if(fabs(localData2[(localId)+i].x) > fabs(localData2[localId].x) && globalId >= colId){
					localData2[localId] = localData2[localId+i];  
				}	 
			}  
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 
			if(i%2 != 0 && i != 1){
				i++;
			}
		}

		if(colId >= workGroupId*256)
			output[workGroupId] = localData2[colId%256];
		else
			output[workGroupId] = localData2[0];
	}else{
		output[workGroupId] = (double2)(0.0, 0.0); 
	}
}	
