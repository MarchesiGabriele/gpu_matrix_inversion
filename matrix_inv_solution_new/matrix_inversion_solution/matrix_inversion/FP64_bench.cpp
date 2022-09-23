﻿#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include "res_struct.h"
#define __CL_ENABLE_EXCEPTIONS

Res FP64_bench(std::vector<double> input_matrix, int matrix_order) {

	std::vector<double> results = {};

	const std::string fixColumnKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void fixColumnKernel(__global double *matrix, int size, int r, __global double *output){

			size_t j = get_global_id(0);	/* column index */
			size_t i = get_global_id(1);	/* row index */
			
			__private double Cij;
			__private double Crj;		
			__private double Cir;

			Cij = matrix[i * size + j];
			Crj = matrix[r * size + j];
			Cir = matrix[i * size + r];

			if(Cir != 0 && i != r){
				Cij = Cij - (Cir * Crj);
			}

			output[i * size + j] = Cij;
		})";

	const std::string maxPivotKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void maxPivotKernel(__global double *matrix, int size, int r, __global double2 *output){
			size_t localSize = get_local_size(0);
			int globalId = get_global_id(0);
			size_t localId = get_local_id(0);
			int workGroupId = get_group_id(0);
			__private int loopLimit = 256;
			__local double2 localData2[256];
			__private int lim = 0;
			localData2[localId] = (double2)(matrix[globalId*size + r], (double)(globalId));		
			barrier(CLK_LOCAL_MEM_FENCE);
			/* check if r is above current workGroup, if it is, no element of the current workGroup can be the max */
			if(r <= (workGroupId*256 + 255)){
				if((size/2) < 256){				
					loopLimit = (int)(size/2);
				}else if(workGroupId == (int)(floor((double)(size/512)))){		/* size == ordine*2 */
					loopLimit = (int)((size/2)%256);
				}
		
				/* Controllo se r è compreso tra gli elementi del workgroup corrente oppure se è sotto, se è sotto non devo tenerne conto durante la ricerca del max */
				if(r>= (workGroupId * 256))
					lim = loopLimit-(r % 256);
				else
					lim = loopLimit;
					
				if(lim%2 != 0){
					localData2[loopLimit] = (double2)(0.0,0.0);
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
				output[workGroupId] = (double2)(0.0, 0.0); 
			}
		})";


	const std::string finalMaxPivotKernelString = R"(
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
		})";



	const std::string fixRowKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void fixRowKernel(__global double *matrix, int size, int r, __global double2 *pivot){
			size_t globalId = get_global_id(0);		

			__private double row;
			__local double Aii;

			row = matrix[size*r+ globalId];
			Aii = pivot[0].x;
			
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 

			matrix[size*r+ globalId] = row/Aii;
		})";

	const std::string pivotKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void pivotElementsKernel(__global double *matrix, int size, int r, __global double2 *pivot){
			size_t globalId = get_global_id(0);
			size_t localId = get_local_id(0);

			__local double localDataOld[256];
			__local double localDataMax[256];
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
		})";

	const std::string matrixKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void makeAugmentedMatrix(__global double *matrix, __global double *inputMatrix, int matrixOrder){
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
		} 
	
		/* La matrice inversa la vado a mettere dentro il buffer della matrice di input. Per evitare di creare un nuovo buffer*/
		__kernel void getInvertedMatrix(__global double *matrix, __global double *inputMatrix, int matrixOrder){
			/* COL VA DA 0 A 2*MATRIX_ORDER */
			int col = get_global_id(0);

			/* ROW VA DA 1 A MATRIX_ORDER */
			int row = get_global_id(1);

			if(col >= matrixOrder){
				inputMatrix[(col-matrixOrder) + (row-1)*matrixOrder] = matrix[col + (row-1)*matrixOrder*2]; 
			}
		}
		)";

	if (matrix_order <= 0) {
		return {};
	}

	int matrix_height = int(input_matrix.size() / matrix_order);
	if (matrix_height != matrix_order) {
		return {};
	}

	try {
		std::vector<cl::Platform>  platforms;
		cl::Platform chosenPlatform;

		std::vector<cl::Device>  devices;
		cl::Device chosenDevice;

		cl::Context context;
		cl::CommandQueue commandQueue;

		cl_int operationResult;

		std::vector<cl_double> augmented_matrix = std::vector<cl_double>(matrix_order * matrix_order * 2, 0);

		using namespace std::chrono;
		steady_clock::time_point tempoTotaleInizio = steady_clock::now();
		steady_clock::time_point tempoComputazioneInizio;
		steady_clock::time_point tempoFixRowInizio;
		steady_clock::time_point tempoFixRowFine;

		operationResult = cl::Platform::get(&platforms);
		std::string platformName;
		std::string platformVendor;
		std::string platformVersion;

		chosenPlatform = platforms[0];

		operationResult = chosenPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		chosenDevice = devices[0];

		context = cl::Context(chosenDevice);

		steady_clock::time_point inizioCq = steady_clock::now();
		commandQueue = cl::CommandQueue(context, chosenDevice, CL_QUEUE_PROFILING_ENABLE);
		steady_clock::time_point fineCq = steady_clock::now();
		duration<float> tempoCq = duration_cast<duration<float>> (fineCq - inizioCq);
		results.push_back(tempoCq.count());

		std::vector<cl::Buffer> buffers;
		steady_clock::time_point inizioCreazioneBuffer = steady_clock::now();
		std::cout << augmented_matrix.size() << std::endl;
		buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, augmented_matrix.size() * sizeof(cl_double), augmented_matrix.data(), &operationResult));

		std::cout << input_matrix.size() << std::endl;
		buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, input_matrix.size() * sizeof(cl_double), input_matrix.data(), &operationResult));

		buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, augmented_matrix.size() * sizeof(cl_double), augmented_matrix.data(), &operationResult));

		int numeroWorkgroups = 0;
		if ((matrix_order % 256) == 0)
			numeroWorkgroups = matrix_order / 256;
		else
			numeroWorkgroups = (int)(matrix_order / 256) + 1;

		std::vector<cl_double2> max_pivots(numeroWorkgroups, cl_double2());
		buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numeroWorkgroups * sizeof(cl_double2), &operationResult));

		steady_clock::time_point fineCreazioneBuffer = steady_clock::now();

		duration<float> tempoCreazioneBuffer = duration_cast<duration<float>> (fineCreazioneBuffer - inizioCreazioneBuffer);
		results.push_back(tempoCreazioneBuffer.count());

		cl::Program fix_column_program(context, cl::Program::Sources(1, std::make_pair(fixColumnKernelString.c_str(), fixColumnKernelString.length() + 1)), &operationResult);
		cl::Program max_pivot_program(context, cl::Program::Sources(1, std::make_pair(maxPivotKernelString.c_str(), maxPivotKernelString.length() + 1)), &operationResult);
		cl::Program final_max_pivot_program(context, cl::Program::Sources(1, std::make_pair(finalMaxPivotKernelString.c_str(), finalMaxPivotKernelString.length() + 1)), &operationResult);
		cl::Program matrix_program(context, cl::Program::Sources(1, std::make_pair(matrixKernelString.c_str(), matrixKernelString.length() + 1)), &operationResult);
		cl::Program fix_row_program(context, cl::Program::Sources(1, std::make_pair(fixRowKernelString.c_str(), fixRowKernelString.length() + 1)), &operationResult);
		cl::Program pivot_kernel_program(context, cl::Program::Sources(1, std::make_pair(pivotKernelString.c_str(), pivotKernelString.length() + 1)), &operationResult);

		steady_clock::time_point inizioCompilazioneProgrammi = steady_clock::now();
		operationResult = fix_column_program.build(devices);
		operationResult = max_pivot_program.build(devices);
		operationResult = final_max_pivot_program.build(devices);
		operationResult = matrix_program.build(devices);
		operationResult = fix_row_program.build(devices);
		operationResult = pivot_kernel_program.build(devices);
		steady_clock::time_point fineCompilazioneProgrammi = steady_clock::now();
		duration<float> tempoCompilazioneProgrammi = duration_cast<duration<float>> (fineCompilazioneProgrammi - inizioCompilazioneProgrammi);
		results.push_back(tempoCompilazioneProgrammi.count());

		cl::Kernel fix_column_kernel(fix_column_program, "fixColumnKernel", &operationResult);
		cl::Kernel max_pivot_kernel(max_pivot_program, "maxPivotKernel", &operationResult);
		cl::Kernel final_max_pivot_kernel(final_max_pivot_program, "finalMaxPivotKernel", &operationResult);
		auto kernelWorkGroupSize = fix_column_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(chosenDevice);
		cl::Kernel make_augmented_matrix_kernel(matrix_program, "makeAugmentedMatrix", &operationResult);
		cl::Kernel get_inverted_matrix_kernel(matrix_program, "getInvertedMatrix", &operationResult);
		cl::Kernel fix_row_kernel(fix_row_program, "fixRowKernel", &operationResult);
		cl::Kernel pivot_kernel(pivot_kernel_program, "pivotElementsKernel", &operationResult);

		steady_clock::time_point inizioMakeAug = steady_clock::now();
		operationResult = make_augmented_matrix_kernel.setArg(0, buffers[0]);
		operationResult = make_augmented_matrix_kernel.setArg(1, buffers[1]);
		operationResult = make_augmented_matrix_kernel.setArg(2, matrix_order);
		operationResult = commandQueue.enqueueNDRangeKernel(make_augmented_matrix_kernel, cl::NDRange(0, 1), cl::NDRange((cl_int)(matrix_order * 2), matrix_order), cl::NullRange, NULL, NULL);

		operationResult = commandQueue.finish();
		steady_clock::time_point fineMakeAug = steady_clock::now();
		duration<float> tempoMakeAug = duration_cast<duration<float>> (fineMakeAug- inizioMakeAug);
		results.push_back(tempoMakeAug.count());

		// MAX PIVOT
		operationResult = max_pivot_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata
		operationResult = max_pivot_kernel.setArg(3, buffers[3]);
		// FINAL MAX PIVOT
		operationResult = final_max_pivot_kernel.setArg(0, buffers[3]);
		// PIVOT 
		operationResult = pivot_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata
		operationResult = pivot_kernel.setArg(3, buffers[3]);
		// ROWS
		operationResult = fix_row_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata
		operationResult = fix_row_kernel.setArg(3, buffers[3]);
		// COLUMNS
		operationResult = fix_column_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata

		tempoComputazioneInizio = steady_clock::now();
		duration<float> pivotComputeTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
		duration<float> pivotTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
		duration<float> rowTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
		double columnTime = 0.0;
		duration<float> readWriteTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());

		std::vector<cl::Event> event = {};
		cl::Event evento;
		event.push_back(evento);

		cl_ulong time_start;
		cl_ulong time_end;
		for (int i = 0; i < matrix_order; i++) {
			bool flag = (i % 2) == 0;

			steady_clock::time_point pivotInizio = steady_clock::now();
			if (flag)
				operationResult = max_pivot_kernel.setArg(0, buffers[0]);
			else
				operationResult = max_pivot_kernel.setArg(0, buffers[2]);

			operationResult = max_pivot_kernel.setArg(2, i);
			operationResult = commandQueue.enqueueNDRangeKernel(max_pivot_kernel, cl::NullRange, cl::NDRange(matrix_order), cl::NDRange(256), NULL, NULL);

			// FINAL MAX PIVOT
			operationResult = commandQueue.enqueueNDRangeKernel(final_max_pivot_kernel, cl::NullRange, cl::NDRange(numeroWorkgroups), cl::NullRange, NULL, NULL);
			
			// PIVOT
			if (flag)
				operationResult = pivot_kernel.setArg(0, buffers[0]);
			else
				operationResult = pivot_kernel.setArg(0, buffers[2]);

			operationResult = pivot_kernel.setArg(2, i); // index riga su cui fare il pivot 
			operationResult = commandQueue.enqueueNDRangeKernel(pivot_kernel, cl::NullRange, cl::NDRange(matrix_order * 2), cl::NDRange(256), NULL, NULL);
			
			commandQueue.finish();
			steady_clock::time_point pivotFine = steady_clock::now();
			pivotTime += duration_cast<duration<float>> (pivotFine - pivotInizio);

			// ROWS
			steady_clock::time_point rowInizio = steady_clock::now();
			if (flag)
				operationResult = fix_row_kernel.setArg(0, buffers[0]);
			else
				operationResult = fix_row_kernel.setArg(0, buffers[2]);

			operationResult = fix_row_kernel.setArg(2, i); // index riga da fixare
			operationResult = commandQueue.enqueueNDRangeKernel(fix_row_kernel, cl::NullRange, cl::NDRange(matrix_order * 2), cl::NDRange(256), NULL, NULL);

			commandQueue.finish();
			steady_clock::time_point rowFine = steady_clock::now();
			rowTime += duration_cast<duration<float>> (rowFine - rowInizio);
			
			// COLUMNS
			if (flag) {
				operationResult = fix_column_kernel.setArg(0, buffers[0]); // read
				operationResult = fix_column_kernel.setArg(3, buffers[2]); // write
			}
			else {
				operationResult = fix_column_kernel.setArg(0, buffers[2]); // read 
				operationResult = fix_column_kernel.setArg(3, buffers[0]); // write
			}
			operationResult = fix_column_kernel.setArg(2, i); // index colonna da fixare
			operationResult = commandQueue.enqueueNDRangeKernel(fix_column_kernel, cl::NullRange, cl::NDRange(matrix_order * 2, matrix_order), cl::NullRange, NULL, &event[0]);

			event[0].waitForEvents;
			commandQueue.finish();


			event[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
			event[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
			columnTime += time_end - time_start;
		}

		operationResult = commandQueue.finish();
		steady_clock::time_point tempoComputazioneFine = steady_clock::now();

		results.push_back(pivotTime.count());
		results.push_back(rowTime.count());
		results.push_back(columnTime/1e9);

		duration<float> tempoComputazioneGPU = duration_cast<duration<float>> (tempoComputazioneFine - tempoComputazioneInizio);
		results.push_back(tempoComputazioneGPU.count());

		// GET INVERTED MATRIX 
		steady_clock::time_point getInvertedInizio = steady_clock::now();
		if (((matrix_order - 1) % 2) == 0)
			operationResult = get_inverted_matrix_kernel.setArg(0, buffers[2]);
		else
			operationResult = get_inverted_matrix_kernel.setArg(0, buffers[0]);

		operationResult = get_inverted_matrix_kernel.setArg(1, buffers[1]);
		operationResult = get_inverted_matrix_kernel.setArg(2, matrix_order);
		operationResult = commandQueue.enqueueNDRangeKernel(get_inverted_matrix_kernel, cl::NDRange(0, 1), cl::NDRange((cl_int)(2 * matrix_order), matrix_order), cl::NullRange, NULL, NULL);
		operationResult = commandQueue.finish();
		steady_clock::time_point getInvertedFine = steady_clock::now();
		duration<float> tempoGetInverted = duration_cast<duration<float>> (getInvertedFine - getInvertedInizio);

		results.push_back(tempoGetInverted.count());



		std::vector<double> matriceResult = std::vector<double>(input_matrix.size(), 0.0);
		operationResult = commandQueue.enqueueReadBuffer(buffers[1], CL_TRUE, 0, matriceResult.size() * sizeof(cl_double), matriceResult.data(), NULL);

		operationResult = commandQueue.finish();

		steady_clock::time_point tempoTotaleFine = steady_clock::now();
		duration<float> tempoTotale = duration_cast<duration<float>> (tempoTotaleFine - tempoTotaleInizio);
		results.push_back(tempoTotale.count());

		buffers.clear();

		Res res;
		res.inversa64 = matriceResult;
		res.times = results;

		return res;
	}
	catch (cl_int e) {
		std::cerr << "ERRORE N°: " << e << std::endl;
	}
	return {};
}



