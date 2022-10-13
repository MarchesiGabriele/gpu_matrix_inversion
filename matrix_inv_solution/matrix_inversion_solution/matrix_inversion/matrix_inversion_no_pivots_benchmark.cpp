#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include "res_struct.h"
#define __CL_ENABLE_EXCEPTIONS

Res matrix_inversion_no_pivots_bench(std::vector<double> input_matrix, int matrix_order) {

	std::vector<double> results = {};


	// KERNEL PER FIXARE COLONNE
	const std::string fixColumnKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void fixColumnKernel(__global double *matrix, int size, int colIdx, __global double *cir){

			size_t globalId = get_global_id(0);		/* column index */
			size_t globalId2 = get_global_id(1);	/* row index */
			
			__private double currentRowValue;
			__private double otherRowValue;
			__private double AiIdx;

			currentRowValue = matrix[globalId2 * size + globalId];
			otherRowValue = matrix[colIdx * size + globalId];
			AiIdx = cir[globalId2];

			if(AiIdx != 0 && globalId2 != colIdx){
				currentRowValue = currentRowValue - (AiIdx * otherRowValue);
				matrix[globalId2*size + globalId] = currentRowValue;
			}

		})";


	// QUESTO KERNEL RECUPERA IL VALORE SULLA DIAGONALE USATO DA FIX ROW
	// La dimensione globale è: numeroWorkGroups.	
	const std::string findCrrKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void findCrr(__global double *Crr, __global double *matrix, int index, int size){
			Crr[0] = matrix[index*size + index];		
		})";

	// COPIO COLONNA DI INDEX R DENTRO UN BUFFER A PARTE. QUESTA COLONNA CONTIENE I VALORI CIR USATI DA FIX COLUMN.
	// "r" indica l'index della colonna che devo copiare
	const std::string copyCirColumnKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void copyCirColumn(__global double *cirColumn, __global double *matrix, int r, int size){
			size_t j = get_global_id(0);		
			cirColumn[j] = matrix[j*size + r];
		})";



	const std::string fixRowKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void fixRowKernel(__global double *matrix, int size, int rowId, __global double *crr){
			size_t globalId = get_global_id(0);		

			__private double row;
			__private double Aii;

			row = matrix[size*rowId + globalId];
			Aii = crr[0];

			matrix[size*rowId + globalId] = row/Aii;
		})";


	// Con matrixOrder si indende l'ordine della matrice iniziale, non la larghezza della matrice augmentata!!
	// Finche sono a sinistra della matrice augmentata, leggo e scrivo dalla matrice input alla matrice augmentata
	// Se sono a destra allora scrivo solamente il valore 1 o 0. 
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



	// First check
	if (matrix_order <= 0) {
		return {};
	}

	// Check if the input matrix is square 
	int matrix_height = int(input_matrix.size() / matrix_order);
	if (matrix_height != matrix_order) {
		return {};
	}

	try {
		// lista delle piattaforme disponibili 
		std::vector<cl::Platform>  platforms;
		cl::Platform chosenPlatform;

		// lista dei device disponibili  
		std::vector<cl::Device>  devices;
		cl::Device chosenDevice;

		// contesto della piattaforma in cui mi trovo
		cl::Context context;

		// Command queue per il kernel ______
		cl::CommandQueue commandQueue;

		// Variabile usata per immagazzinare i risultati/errori delle operazioni che vengono eseguite
		cl_int operationResult;

		std::string platformName;
		std::string platformVendor;
		std::string platformVersion;

		// primo parametro funzione -> matrice da invertire (sofforma di vettore o vettore di vettori)
		std::vector<cl_double> matrice_input = input_matrix;

		// Ordine Matrice  
		// TODO CONTROLLARE  CHE LA MATRICE INSERITA SIA  QUADRATA !!
		cl_int matrix_order = sqrt(matrice_input.size());

		// matrice input a sinistra e matrice identita a destra (n x 2n)
		// la dimensione di questa matrice � la dimensione globale
		std::vector<cl_double> matrice_augmentata = std::vector<cl_double>((cl_double)matrix_order * matrix_order * 2, 0);

		using namespace std::chrono;
		// Tempo totale impiegato per eseguire la funzione matrix_inversio() 
		steady_clock::time_point tempoTotaleInizio = steady_clock::now();
		// Tempo impiegato dalla GPU per eseguire i kernel
		steady_clock::time_point tempoComputazioneInizio;
		// TEmpo impiegato dal fix row kernel. Usato per calcolare la bandwidth
		steady_clock::time_point tempoFixRowInizio;
		steady_clock::time_point tempoFixRowFine;

		// Recupero le piattaforme disponibili
		operationResult = cl::Platform::get(&platforms);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GETTING PLATFORM" << std::endl;
			throw operationResult;
		}

		// Scelgo piattaforma AMD 
		chosenPlatform = platforms[0];

		// Recupero i device disponibili e mostro INFO DEVICE
		// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html
		operationResult = chosenPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GETTING DEVICES" << std::endl;
			throw operationResult;
		}

		// Scelgo scheda grafica come DEVICE
		chosenDevice = devices[0];

		// Creo il context per il device scelto
		context = cl::Context(chosenDevice);

		// Creo la command queue per il device scelto
		steady_clock::time_point inizioCq = steady_clock::now();
		commandQueue = cl::CommandQueue(context, chosenDevice, CL_QUEUE_PROFILING_ENABLE);
		steady_clock::time_point fineCq = steady_clock::now();
		duration<float> tempoCq = duration_cast<duration<float>> (fineCq - inizioCq);
		results.push_back(tempoCq.count());

		std::vector<cl::Buffer> buffers;
		// 0 Creo buffers n x 2n per matrice augmentata
		steady_clock::time_point inizioCreazioneBuffer = steady_clock::now();
		buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrice_augmentata.size() * sizeof(cl_double), matrice_augmentata.data(), &operationResult));

		// 1 Creo buffer per matrice input
		buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrice_input.size() * sizeof(cl_double), matrice_input.data(), &operationResult));

		// 2 Buffer buffer contenente i Cir.  
		std::vector<cl_double> cir(matrix_order, 0);
		buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrix_order * sizeof(cl_double), cir.data(), &operationResult));

		// 3 Buffer per Crr 
		std::vector<cl_double> crr(1, 0);
		buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double), crr.data(), &operationResult));

		steady_clock::time_point fineCreazioneBuffer = steady_clock::now();
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING BUFFERS1" << std::endl;
			throw operationResult;
		}
		duration<float> tempoCreazioneBuffer = duration_cast<duration<float>> (fineCreazioneBuffer - inizioCreazioneBuffer);
		results.push_back(tempoCreazioneBuffer.count());


		// Creo programmi usando i kernel
		cl::Program fix_column_program(context, cl::Program::Sources(1, std::make_pair(fixColumnKernelString.c_str(), fixColumnKernelString.length() + 1)), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING PROGRAM FIX COLUMNS" << std::endl;
			throw operationResult;
		}

		cl::Program find_crr_program(context, cl::Program::Sources(1, std::make_pair(findCrrKernelString.c_str(), findCrrKernelString.length() + 1)), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING PROGRAM FIND CRR" << std::endl;
			throw operationResult;
		}

		cl::Program copy_cir_column_program(context, cl::Program::Sources(1, std::make_pair(copyCirColumnKernelString.c_str(), copyCirColumnKernelString.length() + 1)), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING COPY CIR COLUMN PROGRAM" << std::endl;
			throw operationResult;
		}

		// Questo programma viene creato e compilato una sola volta, poi viene utilizzato per creare due kernel diversi. Quello per creare la matrice augmentata
		// e quello per estrarre la matrice inversa dalla matrice augmentata.
		cl::Program matrix_program(context, cl::Program::Sources(1, std::make_pair(matrixKernelString.c_str(), matrixKernelString.length() + 1)), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING MATRIX PROGRAM " << std::endl;
			throw operationResult;
		}

		cl::Program fix_row_program(context, cl::Program::Sources(1, std::make_pair(fixRowKernelString.c_str(), fixRowKernelString.length() + 1)), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING PROGRAM FIX ROWS" << std::endl;
			throw operationResult;
		}


		///////////////////////////////////////////////////////////////
		/// Compilo i programmi
		steady_clock::time_point inizioCompilazioneProgrammi = steady_clock::now();
		operationResult = fix_column_program.build(devices);
		if (operationResult == CL_BUILD_PROGRAM_FAILURE) {
			std::string err;
			fix_column_program.getBuildInfo(chosenDevice, CL_PROGRAM_BUILD_LOG, &err);
			std::cout << err;
		}
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR BUILDING PROGRAM FIX COLUMNS" << std::endl;
			throw operationResult;
		}

		operationResult = find_crr_program.build(devices);
		if (operationResult == CL_BUILD_PROGRAM_FAILURE) {
			std::string err;
			find_crr_program.getBuildInfo(chosenDevice, CL_PROGRAM_BUILD_LOG, &err);
			std::cout << err;
		}
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR BUILDING PROGRAM FIND CRR" << std::endl;
			throw operationResult;
		}

		operationResult = copy_cir_column_program.build(devices);
		if (operationResult == CL_BUILD_PROGRAM_FAILURE) {
			std::string err;
			copy_cir_column_program.getBuildInfo(chosenDevice, CL_PROGRAM_BUILD_LOG, &err);
			std::cout << err;
		}
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR BUILDING PROGRAM COPY CIR COLUMN" << std::endl;
			throw operationResult;
		}

		operationResult = matrix_program.build(devices);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR BUILDING MATRIX PROGRAM" << std::endl;
			throw operationResult;
		}
		if (operationResult == CL_BUILD_PROGRAM_FAILURE) {
			std::string err;
			matrix_program.getBuildInfo(chosenDevice, CL_PROGRAM_BUILD_LOG, &err);
			std::cout << err;
		}

		operationResult = fix_row_program.build(devices);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR BUILDING PROGRAM FIX ROWS" << std::endl;
			throw operationResult;
		}
		if (operationResult == CL_BUILD_PROGRAM_FAILURE) {
			std::string err;
			fix_row_program.getBuildInfo(chosenDevice, CL_PROGRAM_BUILD_LOG, &err);
			std::cout << err;
		}

		steady_clock::time_point fineCompilazioneProgrammi = steady_clock::now();
		duration<float> tempoCompilazioneProgrammi = duration_cast<duration<float>> (fineCompilazioneProgrammi - inizioCompilazioneProgrammi);
		results.push_back(tempoCompilazioneProgrammi.count());

		///////////////////////////////////////////////////////////////
		/// Creo i kernel
		cl::Kernel fix_column_kernel(fix_column_program, "fixColumnKernel", &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING FIX COLUMN KERNEL" << std::endl;
			throw operationResult;
		}

		cl::Kernel find_crr_kernel(find_crr_program, "findCrr", &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING FIND CRR KERNEL" << std::endl;
			throw operationResult;
		}

		cl::Kernel copy_cir_column_kernel(copy_cir_column_program, "copyCirColumn", &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING COPY CIR COLUMN KERNEL" << std::endl;
			throw operationResult;
		}

		cl::Kernel make_augmented_matrix_kernel(matrix_program, "makeAugmentedMatrix", &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING MAKE AUGMENTED MATRIX KERNEL" << std::endl;
			throw operationResult;
		}

		cl::Kernel get_inverted_matrix_kernel(matrix_program, "getInvertedMatrix", &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING GET INVERTED MATRIX KERNEL" << std::endl;
			throw operationResult;
		}

		cl::Kernel fix_row_kernel(fix_row_program, "fixRowKernel", &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING FIX ROW KERNEL" << std::endl;
			throw operationResult;
		}

		/// Imposto argomenti kernel + esecuzione kernel 
		///
		/// Con dei cicli itero attraverso le colonne/righe della matrice augmentata
		/// Ad ogni ciclo imposto nuovi parametri al kernel e procedo con una nuova esecuzione
		// Ci pensa openCL ad aspettare che un  kernel finisca prima di inizaire l'altro
		// MAKE AUGMENTED MATRIX
		steady_clock::time_point inizioMatrAug = steady_clock::now();
		operationResult = make_augmented_matrix_kernel.setArg(0, buffers[0]);
		operationResult = make_augmented_matrix_kernel.setArg(1, buffers[1]);
		operationResult = make_augmented_matrix_kernel.setArg(2, matrix_order);
		operationResult = commandQueue.enqueueNDRangeKernel(make_augmented_matrix_kernel, cl::NDRange(0, 1), cl::NDRange((cl_int)(matrix_order * 2), matrix_order), cl::NullRange, NULL, NULL);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR MAKE AUGMENTED KERNEL EXECUTION" << std::endl;
			throw operationResult;
		}
		std::vector<cl_double> m = std::vector<cl_double>(matrice_augmentata.size(), 0);

		operationResult = commandQueue.finish();
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GETTING DEVICES" << std::endl;
			throw operationResult;
		}
		steady_clock::time_point fineMatrAug = steady_clock::now();
		duration<float> tempoMatrAug = duration_cast<duration<float>> (fineMatrAug - inizioMatrAug);
		results.push_back(tempoMatrAug.count());


		// FIND CRR 
		operationResult = find_crr_kernel.setArg(0, buffers[3]);
		operationResult = find_crr_kernel.setArg(1, buffers[0]);
		operationResult = find_crr_kernel.setArg(3, matrix_order * 2);

		// ROWS
		operationResult = fix_row_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata
		operationResult = fix_row_kernel.setArg(3, buffers[3]);
		operationResult = fix_row_kernel.setArg(0, buffers[0]);

		// COPY CIR COLUMN 
		operationResult = copy_cir_column_kernel.setArg(0, buffers[2]);
		operationResult = copy_cir_column_kernel.setArg(1, buffers[0]);
		operationResult = copy_cir_column_kernel.setArg(3, matrix_order * 2);

		// COLUMNS
		operationResult = fix_column_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata
		operationResult = fix_column_kernel.setArg(0, buffers[0]);
		operationResult = fix_column_kernel.setArg(3, buffers[2]);

		tempoComputazioneInizio = steady_clock::now();
		duration<float> pivotComputeTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
		duration<float> copyTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
		duration<float> crrTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
		duration<float> rowTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
		//duration<float> columnTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
		double columnTime = 0.0;
		duration<float> readWriteTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());

		std::vector<cl::Event> event;
		cl::Event evento;
		event.push_back(evento);

		cl_ulong time_start;
		cl_ulong time_end;
		for (int r = 0; r < matrix_order; r++) {
			// FIND CRR 
			steady_clock::time_point crrInizio = steady_clock::now();
			operationResult = find_crr_kernel.setArg(2, r);
			operationResult = commandQueue.enqueueNDRangeKernel(find_crr_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, NULL);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR FINAL MAX PIVOT KERNEL" << std::endl;
				throw operationResult;
			}
			//////////////////////////////////////////////////////////////////////
			commandQueue.finish();
			//////////////////////////////////////////////////////////////////////
			steady_clock::time_point crrFine = steady_clock::now();
			crrTime += duration_cast<duration<float>> (crrFine - crrInizio);

			// ROWS
			steady_clock::time_point rowInizio = steady_clock::now();

			operationResult = fix_row_kernel.setArg(2, r); // index riga da fixare
			operationResult = commandQueue.enqueueNDRangeKernel(fix_row_kernel, cl::NDRange(r), cl::NDRange(matrix_order + 1), cl::NDRange(256), NULL, NULL);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR KERNEL ROW" << std::endl;
				throw operationResult;
			}

			//////////////////////////////////////////////////////////////////////
			commandQueue.finish();
			//////////////////////////////////////////////////////////////////////
			steady_clock::time_point rowFine = steady_clock::now();
			rowTime += duration_cast<duration<float>> (rowFine - rowInizio);
		

			// COPY CIR COLUMN
			steady_clock::time_point copyInizio = steady_clock::now();

			operationResult = copy_cir_column_kernel.setArg(2, r); // index riga da fixare
			operationResult = commandQueue.enqueueNDRangeKernel(copy_cir_column_kernel, cl::NullRange, cl::NDRange(matrix_order), cl::NullRange, NULL, NULL);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR KERNEL ROW" << std::endl;
				throw operationResult;
			}

			//////////////////////////////////////////////////////////////////////
			commandQueue.finish();
			//////////////////////////////////////////////////////////////////////
			steady_clock::time_point copyFine = steady_clock::now();
			copyTime += duration_cast<duration<float>> (copyFine - copyInizio);

			// COLUMNS
			operationResult = fix_column_kernel.setArg(2, r); // index colonna da fixare
			operationResult = commandQueue.enqueueNDRangeKernel(fix_column_kernel, cl::NDRange(r, 0), cl::NDRange(matrix_order + 1, matrix_order), cl::NullRange, NULL, &event[0]);
			//operationResult = commandQueue.enqueueNDRangeKernel(fix_column_kernel, cl::NDRange(r, 0), cl::NDRange(matrix_order + 1, matrix_order), cl::NullRange, NULL, NULL);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR FIX COLUMNs KERNEL" << std::endl;
				throw operationResult;
			}

			//////////////////////////////////////////////////////////////////////
			//*
			event[0].waitForEvents;
			commandQueue.finish();
			event[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
			event[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
			columnTime += time_end - time_start;
			//*/
			//////////////////////////////////////////////////////////////////////
			
		}

		operationResult = commandQueue.finish();
		steady_clock::time_point tempoComputazioneFine = steady_clock::now();

		
		results.push_back(crrTime.count());
		results.push_back(rowTime.count());
		results.push_back(copyTime.count());
		results.push_back(columnTime/1e9);

		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GETTING DEVICES" << std::endl;
			throw operationResult;
		}

		// TODO: TENERE QUESTO READ BUFFER PERCHé SERVER PER LA LETTURA DEL RISULTATO 
		operationResult = commandQueue.enqueueReadBuffer(buffers[0], CL_TRUE, 0, matrice_augmentata.size() * sizeof(cl_double), m.data(), NULL);

		operationResult = commandQueue.finish();
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GETTING DEVICES" << std::endl;
			throw operationResult;
		}

		duration<float> tempoComputazioneGPU = duration_cast<duration<float>> (tempoComputazioneFine - tempoComputazioneInizio);
		results.push_back(tempoComputazioneGPU.count());

		// GET INVERTED MATRIX 
		steady_clock::time_point getInvertedInizio = steady_clock::now();
		operationResult = get_inverted_matrix_kernel.setArg(0, buffers[0]);

		operationResult = get_inverted_matrix_kernel.setArg(1, buffers[1]);
		operationResult = get_inverted_matrix_kernel.setArg(2, matrix_order);

		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GETTING DEVICES" << std::endl;
			throw operationResult;
		}

		operationResult = commandQueue.enqueueNDRangeKernel(get_inverted_matrix_kernel, cl::NDRange(0, 1), cl::NDRange((cl_int)(2 * matrix_order), matrix_order), cl::NullRange, NULL, NULL);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GET INVERTED MATRIX KERNEL EXECUTION" << std::endl;
			throw operationResult;
		}
		
		commandQueue.finish();
		steady_clock::time_point getInvertedFine = steady_clock::now();
		duration<float> tempoGetInverted = duration_cast<duration<float>> (getInvertedFine - getInvertedInizio);
		results.push_back(tempoGetInverted.count());

		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GETTING DEVICES" << std::endl;
			throw operationResult;
		}



		std::vector<cl_double> matriceResult = std::vector<cl_double>(matrice_input.size(), 0.0);
		operationResult = commandQueue.enqueueReadBuffer(buffers[1], CL_TRUE, 0, matriceResult.size() * sizeof(cl_double), matriceResult.data(), NULL);

		operationResult = commandQueue.finish();
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GETTING DEVICES" << std::endl;
			throw operationResult;
		}

		steady_clock::time_point tempoTotaleFine = steady_clock::now();
		duration<float> tempoTotale = duration_cast<duration<float>> (tempoTotaleFine - tempoTotaleInizio);
		results.push_back(tempoTotale.count());


		buffers.clear();

		Res risultato;
		risultato.inversa = matriceResult;
		risultato.times = results;

		return risultato;
	}
	catch (cl_int e) {
		std::cerr << "ERRORE N°: " << e << std::endl;
	}
	return {};
}




