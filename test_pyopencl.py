import pyopencl as cl
import numpy as np
import os
import math
import time
import warnings

warnings.filterwarnings("ignore")

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0'

N = 4096 
REP = 1 

def matrix_inv():
    # CREO MATRICI INPUT
    matrice_input= np.random.uniform(0, 100, (N,N)).astype(np.float32)
    matrice_input2= matrice_input.copy()
    #print(matrice_input)


    # INIZIALIZZO CONTEXT e COMMANQUEUE
    st7 = time.monotonic()
    st1 = time.monotonic()
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    end1 = time.monotonic()


    # BUFFERS
    st2 = time.monotonic()
    mf = cl.mem_flags
    matrice_augmentata_buf = cl.Buffer(ctx, mf.READ_WRITE, size = matrice_input.nbytes*2)
    matrice_augmentata2_buf = cl.Buffer(ctx, mf.READ_WRITE, size = matrice_input.nbytes*2)
    matrice_input_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = matrice_input)
    n_workgroups = (N // 256)+1
    pivot_parziali_buf = cl.Buffer(ctx, mf.READ_WRITE, n_workgroups*np.dtype(cl.cltypes.float2).itemsize)
    end2 = time.monotonic()

    # KERNELS
    st3 = time.monotonic()
    fix_col_prg = cl.Program(ctx, """
            __kernel void fixColumn(__global float *matrix, int size, int r, __global float *output, int remain, int limit){

                int j = get_global_id(0)*4;
                size_t i = get_global_id(1);
                
                __private float4 Cij;
                __private float4 Crj;		
                __private float Cir;

                // Questi valori precedenti a r sono sempre 0, quindi evito di leggere e fare i calcoli
                if(j+3 < r && (j/4) != (limit-1)){
                    output[i * size + j] = 0;
                    output[i * size + j+1] = 0;
                    output[i * size + j+2] = 0;
                    output[i * size + j+3] = 0;
                    return;
                }


                Cir = matrix[i * size + r];

                Cij.x = matrix[i * size + j];
                Cij.y = matrix[i * size + j+1];
                Cij.z = matrix[i * size + j+2];
                Cij.w = matrix[i * size + j+3];

                if(Cir != 0 && i != r){
                    Crj.x = matrix[r * size + j];
                    Crj.y = matrix[r * size + j+1];
                    Crj.z = matrix[r * size + j+2];
                    Crj.w = matrix[r * size + j+3];

                    Cij.x = Cij.x - (Cir * Crj.x);
                    Cij.y = Cij.y - (Cir * Crj.y);
                    Cij.z = Cij.z - (Cir * Crj.z);
                    Cij.w = Cij.w - (Cir * Crj.w);
                }

                output[i * size + j] = Cij.x;
                output[i * size + j+1] = Cij.y;
                output[i * size + j+2] = Cij.z;
                output[i * size + j+3] = Cij.w;

                if(remain != 0 && (j/4) == (limit-1)){
                    for(int s = 0; s < remain; s++){
                        float crj = matrix[r * size + j+4+s];
                        float cij = matrix[i * size + j+4+s];

                        if(Cir != 0 && i != r){
                            cij = cij - (Cir * crj);
                        }

                        output[i * size + j+4+s] = cij;
                   }
                }
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
            __kernel void maxPivot(__global float *matrix, int size, int r, __global float2 *output){
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

    end3 = time.monotonic()

    # MAKE AUGMENTED MATRIX

    st4 = time.monotonic()
    mam = make_augmented_matrix_prg.makeAugmentedMatrix
    mam.set_args(matrice_augmentata_buf, matrice_input_buf, np.int32(N))
    res = cl.enqueue_nd_range_kernel(queue, mam, [N*2, N], None, global_work_offset = [0,1])
    queue.finish()
    end4 = time.monotonic()

    fr = fix_row_prg.fixRow
    fc = fix_col_prg.fixColumn
    mp = max_pivot_prg.maxPivot
    fmp = final_max_pivot_prg.finalMaxPivot
    p = pivot_prg.pivot

    # KERNEL LOOP
    st5 = time.monotonic()

    pp = (N*2)%4
    for r in range(N): 
        flag = (r%2) == 0

        
        # MAX PIVOT 
        if flag:
            mp.set_args(matrice_augmentata_buf, np.int32(N*2), np.int32(r), pivot_parziali_buf) 
        else:
            mp.set_args(matrice_augmentata2_buf, np.int32(N*2), np.int32(r), pivot_parziali_buf) 

        res = cl.enqueue_nd_range_kernel(queue, mp, [N], [256])

        #FINAL MAX PIVOT
        fmp.set_args(pivot_parziali_buf) 
        res = cl.enqueue_nd_range_kernel(queue, fmp, [n_workgroups], None)


        #PIVOT
        if flag:
            p.set_args(matrice_augmentata_buf, np.int32(N*2), np.int32(r), pivot_parziali_buf) 
        else:
            p.set_args(matrice_augmentata2_buf, np.int32(N*2), np.int32(r), pivot_parziali_buf) 

        res = cl.enqueue_nd_range_kernel(queue, p, [N*2], [256])


        #ROW
        if flag:
            fr.set_args(matrice_augmentata_buf, np.int32(N*2), np.int32(r), pivot_parziali_buf) 
        else:
            fr.set_args(matrice_augmentata2_buf, np.int32(N*2), np.int32(r), pivot_parziali_buf) 

        res = cl.enqueue_nd_range_kernel(queue, fr, [N*2], [256])


        #COLUMN
        # TODO: PROVARE AD USARE GLI OFFSET, solo offset iniziale
        if flag:
            fc.set_args(matrice_augmentata_buf, np.int32(N*2), np.int32(r), matrice_augmentata2_buf, np.int32(pp), np.int32((N*2)//4)) 
        else:
            fc.set_args(matrice_augmentata2_buf, np.int32(N*2), np.int32(r), matrice_augmentata_buf, np.int32(pp), np.int32((N*2)//4))
        res = cl.enqueue_nd_range_kernel(queue, fc, [(N*2)//4, N], None)


    queue.finish()
    end5 = time.monotonic()

    # GET INVERTED MATRIX 
    st6 = time.monotonic()
    gim = get_inverted_matrix_prg.getInvertedMatrix

    if (N-1)%2 == 0:
        gim.set_args(matrice_augmentata2_buf, matrice_input_buf, np.int32(N))
    else:
        gim.set_args(matrice_augmentata_buf, matrice_input_buf, np.int32(N))

    res = cl.enqueue_nd_range_kernel(queue, gim, [N*2, N], None, global_work_offset = [0,1])
    queue.finish()
    end6 = time.monotonic()

    # COPY MATRIX FROM BUFFER TO HOST ARRAY 
    cl.enqueue_copy(queue, matrice_input, matrice_input_buf)
    queue.finish()
    end7 = time.monotonic()

    print("INIZIO CONTROLLO\n")
    print("Tempo queue + context: ", (end1 - st1))
    print("Tempo creazione buffers: ", (end2 - st2))
    print("Tempo build: ", (end3 - st3))
    print("Tempo makeaugmented: ", (end4 - st4))
    print("Tempo computazione: ", (end5 - st5))
    print("Tempo get inverted: ", (end6 - st6))
    print("Tempo totale: ", (end7 - st7), "\n\n")

    # CONTROLLO FINALE
    c = np.matmul(matrice_input, matrice_input2)

    sum = 0
    vec = c.flat
    for i in range(c.size):
        sum += vec[i]*vec[i]

    #print("ordine: ", math.sqrt(N))
    #print("somma: ", math.sqrt(sum))
    #print("errore: ", math.sqrt(N) - math.sqrt(sum))
    return math.sqrt(N) - math.sqrt(sum)


if __name__ == "__main__":
    err = 0
    for h in range(REP):
        err += matrix_inv()

    print(f"Errore medio: {err/REP}")
        



