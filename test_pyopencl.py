
import pyopencl as cl
import numpy as np
import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0'

N = 10 

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

matrice_input= np.random.uniform(0.000001, 100, (N,N)).astype(np.float32)


# BUFFERS
mf = cl.mem_flags
matrice_augmentata_buf = cl.Buffer(ctx, mf.READ_WRITE, size = matrice_input.nbytes*2)
matrice_augmentata2_buf = cl.Buffer(ctx, mf.READ_WRITE, size = matrice_input.nbytes*2)
matrice_input_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = matrice_input)
pivot_parziali_buf = cl.Buffer(ctx, mf.READ_WRITE, ((N // 256)+1)*np.dtype(np.float32).itemsize)

# KERNELS
fix_col_prg = cl.Program(ctx, """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void fixColumn(__global float *matrix, int size, int r, __global float *output){

			size_t j = get_global_id(0);	/* column index */
			size_t i = get_global_id(1);	/* row index */
			
			__private float Cij;
			__private float Crj;		
			__private float Cir;

			Cij = matrix[i * size + j];
			Crj = matrix[r * size + j];
			Cir = matrix[i * size + r];

			if(Cir != 0 && i != r){
				Cij = Cij - (Cir * Crj);
			}

			output[i * size + j] = Cij;
		}"""
        ).build()


fix_row_prg = cl.Program(ctx, """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void fixRow(__global float *matrix, int size, int r, __global float2 *pivot){
			size_t globalId = get_global_id(0);		

			__private float row;
			__local float Aii;

			row = matrix[size*r+ globalId];
			Aii = pivot[0].x;
			
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 

			matrix[size*r+ globalId] = row/Aii;
		}"""
        ).build()

final_max_pivot_prg = cl.Program(ctx, """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void finalMaxPivot(__global float2 *values){
			size_t globalId = get_global_id(0);		
			size_t size = get_global_size(0);		

			__local float2 vector[1000];
			vector[globalId] = values[globalId];

			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		
			if(globalId == 0){
				__private float2 max = (float2)(0.0,0.0);
				for(int i = 0; i<size; i++){
					if(fabs(vector[i].x) > fabs(max.x)){
						max.x = vector[i].x;
						max.y = vector[i].y;
					}
				}
				values[0].x = max.x;
				values[0].y = max.y;
			}
		} """
        ).build()

pivot_prg = cl.Program(ctx, """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void pivot(__global float *matrix, int size, int r, __global float2 *pivot){
			size_t globalId = get_global_id(0);
			size_t localId = get_local_id(0);

			__local float localDataOld[256];
			__local float localDataMax[256];
			__local int maxRow;
		
			maxRow = (int)(pivot[0].y);
			barrier(CLK_LOCAL_MEM_FENCE); 

			localDataOld[localId] = matrix[size*r+ globalId];
			localDataMax[localId] = matrix[size*maxRow + globalId];
			barrier(CLK_LOCAL_MEM_FENCE); 

			if(maxRow != r){
				matrix[size*r+ globalId] = localDataMax[localId];
				matrix[size*maxRow + globalId] = localDataOld[localId];
			}
		}"""
        ).build()

   
make_augmented_matrix_prg = cl.Program(ctx, """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void makeAugmentedMatrix(__global float *matrix, __global float *inputMatrix, int matrixOrder){
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
		}"""
        ).build()



get_inverted_matrix_prg = cl.Program(ctx, """ 
		__kernel void getInvertedMatrix(__global float *matrix, __global float *inputMatrix, int matrixOrder){
			/* COL VA DA 0 A 2*MATRIX_ORDER */
			int col = get_global_id(0);

			/* ROW VA DA 1 A MATRIX_ORDER */
			int row = get_global_id(1);

			if(col >= matrixOrder){
				inputMatrix[(col-matrixOrder) + (row-1)*matrixOrder] = matrix[col + (row-1)*matrixOrder*2]; 
			}
		}"""
        ).build()

max_pivot_prg = cl.Program(ctx, """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void maxPivotKernel(__global float *matrix, int size, int r, __global float2 *output){
			size_t localSize = get_local_size(0);
			int globalId = get_global_id(0);
			size_t localId = get_local_id(0);
			int workGroupId = get_group_id(0);
			__private int loopLimit = 256;
			__local float2 localData2[256];
			__private int lim = 0;
			localData2[localId] = (float2)(matrix[globalId*size + r], (float)(globalId));		
			barrier(CLK_LOCAL_MEM_FENCE);
			if(r <= (workGroupId*256 + 255)){
				if((size/2) < 256){				
					loopLimit = (int)(size/2);
				}else if(workGroupId == (int)(floor((float)(size/512)))){
					loopLimit = (int)((size/2)%256);
				}
		
				if(r>= (workGroupId * 256))
					lim = loopLimit-(r % 256);
				else
					lim = loopLimit;
					
				if(lim%2 != 0){
					localData2[loopLimit] = (float2)(0.0,0.0);
					lim++;
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			
				for(int i = lim >> 1; i > 0; i>>=1){
					if(localId < i && fabs(localData2[(localId)+i].x) > fabs(localData2[localId].x) && globalId >= r){
						localData2[localId] = localData2[localId+i];  
					}	 
					barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 
					if(i%2 != 0 && i != 1){
						i++;
					}
				}
				if(r>= workGroupId*256)
					output[workGroupId] = localData2[r%256];
				else
					output[workGroupId] = localData2[0];
			}else{
				output[workGroupId] = (float2)(0.0, 0.0); 
			}
		}"""
        ).build()


# MAKE AUGMENTED MATRIX
mam = make_augmented_matrix_prg.makeAugmentedMatrix


mam.set_args(matrice_augmentata_buf, matrice_input_buf, np.int32(N))


#res = cl.enqueue_nd_range_kernel(queue, mam, [N*2, N], None, matrice_augmentata_buf, matrice_input_buf, N, global_work_offset = [0,1])





