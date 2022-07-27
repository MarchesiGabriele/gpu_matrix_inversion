#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>
#define __CL_ENABLE_EXCEPTIONS
#define MAX_WORK_GROUP_SIZE 256

/// Prendo matrice in input, calcolo max pivot per ciascuna colonna ed eseguo swap rows.
// matrix_vector � gi� la matrice augmentata
void pivot_max_test(std::vector<double> matrix_vector, int matrix_order) {
	const std::string maxKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void maxKernel(__global double *matrix, int size, int colId, __local double *localData, __global double2 *output){
			size_t localId = get_local_id(0);
			size_t workGroupId = get_group_id(0);
			size_t localSize = get_local_size(0);
			size_t globalId = get_global_id(0);		/* n-i elementi. va da i a n-i */
			
			localData[localId] = matrix[globalId*size + colId];		/* 256 dati del workgroup tra cui devo trovare il max */

			barrier(CLK_LOCAL_MEM_FENCE);
		
			__private bool isMax = true;
			/* Per ciascun workitem, itero tutti i 256 valori per sapere se il valore associato al workitem � quello pi� grande */
			for(int i = 0; i<localSize; i++){
				if((i*workGroupId) > colId && localData[localId] < localData[i]){
					isMax = false;
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			if(isMax){
				__private double2 max = (double2)(0.0, 0.0);
				max.x = localData[localId];
				max.y = localId*workGroupId;
				output[workGroupId] = max;
			}
		})";
	
	// Index indica la riga che contiene il pivot max e che deve essere swappata con la riga indicata da colId
	// Anche per questo kernel ogni work group si occupa di swappare 256 elementi da una riga all'altra
	const std::string swapKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void swapKernel(__global double *matrix, int size, int colId, int index){
			size_t localId = get_local_id(0);
			size_t globalId = get_global_id(0);	
	
			/* riga che contiene il max */	
			__local	double maxRow[256]; 
			maxRow[localId] = matrix[index*size + globalId];

			__local	double currentRow[256];
			currentRow[localId] = matrix[colId*size + globalId];

			barrier(CLK_LOCAL_MEM_FENCE);	

			matrix[index*size + globalId] = currentRow[localId];
			matrix[colId*size + globalId] = maxRow[localId];
		})";

	try {
		std::vector<cl::Platform>  platforms;
		cl::Platform::get(&platforms);
		cl::Platform chosenPlatform = platforms[0];


		std::vector<cl::Device>  devices;
		chosenPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		cl::Device chosenDevice = devices[0];


		cl::Context context;
		context = cl::Context(chosenDevice);


		cl::CommandQueue commandQueue;
		commandQueue = cl::CommandQueue(context, chosenDevice);


		// Contiene matrice iniziale
		// Lo swap viene eseguito sempre sulla matrice iniziale, non uno un buffer diverso solo per l'output
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, matrix_vector.size() * sizeof(cl_double), matrix_vector.data());
		auto workGroupsNumber = matrix_vector.size() / MAX_WORK_GROUP_SIZE;
		// Contiene i risultati di ciascun workgroup
		cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, workGroupsNumber * sizeof(cl_double2));


		cl::Program maxProgram(context, cl::Program::Sources(1, std::make_pair(maxKernelString.c_str(), maxKernelString.length() + 1)));
		cl::Program swapProgram(context, cl::Program::Sources(1, std::make_pair(swapKernelString.c_str(), swapKernelString.length() + 1)));


		cl_int operationResult = maxProgram.build(devices);
		if (operationResult == CL_BUILD_PROGRAM_FAILURE) {
			std::string err;
			maxProgram.getBuildInfo(chosenDevice, CL_PROGRAM_BUILD_LOG, &err);
			std::cout << err;
		}
		operationResult = swapProgram.build(devices);
		if (operationResult == CL_BUILD_PROGRAM_FAILURE) {
			std::string err;
			swapProgram.getBuildInfo(chosenDevice, CL_PROGRAM_BUILD_LOG, &err);
			std::cout << err;
		}


		cl::Kernel maxKernel(maxProgram, "maxKernel");
		cl::Kernel swapKernel(swapProgram, "swapKernel");


		maxKernel.setArg(0, inputBuffer);
		maxKernel.setArg(1, matrix_order * 2);
		maxKernel.setArg(3, MAX_WORK_GROUP_SIZE * sizeof(double));
		maxKernel.setArg(4, outputBuffer);

		std::vector<cl_double2> max = {};
		int indexMax = 0;
		for (int i = 0; i < matrix_order; i++) {
			maxKernel.setArg(2, i);
			commandQueue.enqueueNDRangeKernel(maxKernel, cl::NDRange(0, 0), cl::NDRange(matrix_order, 0), cl::NDRange(256, 0), NULL, NULL);


			commandQueue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, workGroupsNumber * sizeof(cl_double2), max.data(), NULL);
			commandQueue.finish();
			indexMax = 0;
			for (int j = 0; j < MAX_WORK_GROUP_SIZE; j++) {
				for (int k = 0; k < MAX_WORK_GROUP_SIZE; k++) {
					if (max[j].x < max[k].x)
						break;
					indexMax = max[j].y;
				}
			}

			commandQueue.enqueueNDRangeKernel(swapKernel, cl::NDRange(0, 0), cl::NDRange(2 * matrix_order, 0), cl::NDRange(256, 0), NULL, NULL);
		}
			
			commandQueue.finish();
			commandQueue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, matrix_vector.size() * 2 * sizeof(double), matrix_vector.data(), NULL);
			commandQueue.finish();

			std::cout << "\n PIVOTED: " << std::endl;
			for (int i = 0; i < matrix_vector.size(); i++) {
				if (i != 0 && (i % (matrix_order*2)) == 0) {
					std::cout << std::endl;	
				}

				std::cout << matrix_vector[i] << "  ";
			}


	
	}
	catch (cl_int e) {
		std::cout << e;
	}


}