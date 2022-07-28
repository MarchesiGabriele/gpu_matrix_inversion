#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#define __CL_ENABLE_EXCEPTIONS
#define A false 

	std::vector<double> matrix_inversion(std::vector<double> matrix_vector, int matrix_order) {
		// KERNEL PER FIXARE COLONNE
		// NB: size deve essere pari alla larghezza della matrice augmentata
		const std::string fixColumnKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void fixColumnKernel(__global double *matrix, int size, int colIdx, __global double *output){
			int j = get_global_id(0);
			int i = get_global_id(1);

			__local double AiIdx;

			AiIdx = matrix[i*size + colIdx];
			barrier(CLK_LOCAL_MEM_FENCE);

			if(matrix[i*size + colIdx] != 0 && i != colIdx){
				output[i*size + j] = matrix[i*size + j] - (AiIdx * matrix[colIdx*size + j]);
			}else{
				output[i*size + j] = matrix[i*size + j];
			}
		})";

		const std::string fixRowKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void fixRowKernel(__global double *matrix, int size, int rowId, double Aii){
			size_t localId = get_local_id(0);
			size_t globalId = get_global_id(0);		

			__local double row[256];

			row[localId] = matrix[size*rowId + globalId];
			barrier(CLK_LOCAL_MEM_FENCE); 

			matrix[size*rowId + globalId] = row[localId]/Aii;
		})";

		// Size corrisponde alla larghezza della matrice augmentata
		// Max row è l'id della riga che deve eseguire lo swap con rowId
		const std::string pivotKernelString = R"(
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		__kernel void pivotElementsKernel(__global double *matrix, int size, int rowId, int maxRow){
			int j = get_global_id(0);
			double old = 0.0;
			old = matrix[size*rowId + j];
			barrier(CLK_LOCAL_MEM_FENCE); 

			matrix[size*rowId + j] = matrix[size*maxRow + j];
			matrix[size*maxRow + j] = old;
		})";

		// Con matrixOrder si indende l'ordine della matrice iniziale, non la larghezza della matrice augmentata!!
		// Finche sono a sinistra della matrice augmentata, leggo e scrivo dalla matrice input alla matrice augmentata
		// Se sono a destra allora scrivo solamente il valore 1 o 0. 
		const std::string matrixKernelString= R"(
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

		
		//std::cout << std::setprecision(5);

		// se altezza vettore � zero ritorno vettore vuoto
		if (matrix_order <= 0) {
			return {};
		}

		// Controllo se la matrice � quadrata, se non lo � ritorno vettore vuoto
		float matrix_height = matrix_vector.size() / matrix_order * 1.0;
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
			std::vector<cl_double> matrice_input = matrix_vector;

			// Ordine Matrice  
			// TODO CONTROLLARE  CHE LA MATRICE INSERITA SIA  QUADRATA !!
			cl_int matrix_order = sqrt(matrice_input.size());

			// matrice input a sinistra e matrice identita a destra (n x 2n)
			// la dimensione di questa matrice � la dimensione globale
			std::vector<cl_double> matrice_augmentata = std::vector<cl_double>((cl_double)matrix_order*matrix_order*2, 0);

			using namespace std::chrono;
			// Tempo totale impiegato per eseguire la funzione matrix_inversio() 
			steady_clock::time_point tempoTotaleInizio = steady_clock::now();
			// Tempo impiegato dalla GPU per eseguire i kernel
			steady_clock::time_point tempoComputazioneInizio;
			// TEmpo impiegato dal fix row kernel. Usato per calcolare la bandwidth
			steady_clock::time_point tempoFixRowInizio;
			steady_clock::time_point tempoFixRowFine;

			// Recupero le piattaforme disponibili
			steady_clock::time_point inizioRecuperoPlat= steady_clock::now();
			operationResult = cl::Platform::get(&platforms);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING PLATFORM" << std::endl;
				throw operationResult;
			}
			std::cout << "Piattaforme disponibili: " << std::endl;
			for (cl::Platform i : platforms) {
				i.getInfo(CL_PLATFORM_NAME, &platformName);
				i.getInfo(CL_PLATFORM_VENDOR, &platformVendor);
				i.getInfo(CL_PLATFORM_VERSION, &platformVersion);
				std::cout << "PIATTAFORMA: " << platformName << std::endl;
				std::cout << "VENDITORE: " << platformVendor << std::endl;
				std::cout << "VERSIONE: " << platformVersion << std::endl;
				std::cout << std::endl;
			}
			steady_clock::time_point fineRecuperoPlat= steady_clock::now();
			duration<float> tempoRecuperoPlat= duration_cast<duration<float>> (fineRecuperoPlat- inizioRecuperoPlat);
			std::cout << "Tempo Recupero Plat: " << tempoRecuperoPlat.count() << " seconds" << std::endl;


			// Scelgo piattaforma AMD 
			chosenPlatform = platforms[0];

			// Recupero i device disponibili e mostro INFO DEVICE
			// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html
			operationResult = chosenPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING DEVICES" << std::endl;
				throw operationResult;
			}


			// DEVICE INFO
			std::string deviceName;
			cl_ulong globalCacheSize;
			cl_ulong localMemSize;
			cl_ulong maxConstBufferSize;
			size_t maxWorkGroupSize;
			cl_uint maxWorkItemDimensions;
			cl_ulong globalMemSize;
			char extension[400];

			std::cout << "Device disponibili: " << std::endl;
			for (cl::Device i : devices) {
				i.getInfo(CL_DEVICE_NAME, &deviceName);
				i.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &globalCacheSize);
				i.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
				i.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemSize);
				i.getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &maxConstBufferSize);
				i.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
				i.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &maxWorkItemDimensions);
				i.getInfo(CL_DEVICE_EXTENSIONS, &extension);
				std::cout << "DEVICE: " << deviceName << std::endl;
				std::cout << "Global Cache size: " << globalCacheSize << " bytes" << std::endl;
				std::cout << "Global mem size: " << globalMemSize  << " bytes" << std::endl;
				std::cout << "Local mem size: " <<localMemSize << " bytes"<< std::endl;
				std::cout << "Max Const Buffer Size: " << maxConstBufferSize << " bytes"<< std::endl;
				std::cout << "Max Work Group Size: " << maxWorkGroupSize << std::endl;
				std::cout << "Max Work Item dimensions: " << maxWorkItemDimensions << std::endl;
				std::cout << "Device Estensions: " << extension << std::endl;
				std::cout << std::endl;
			}


			// Scelgo scheda grafica come DEVICE
			chosenDevice = devices[0];

			// Creo il context per il device scelto
			steady_clock::time_point inizioCtx= steady_clock::now();
			context = cl::Context(chosenDevice);
			steady_clock::time_point fineCtx= steady_clock::now();
			duration<float> tempoCtx= duration_cast<duration<float>> (fineCtx- inizioCtx);
			std::cout << "Tempo Context" << tempoCtx.count() << " seconds" << std::endl;

			// Creo la command queue per il device scelto
			steady_clock::time_point inizioCq= steady_clock::now();
			commandQueue = cl::CommandQueue(context, chosenDevice);			
			steady_clock::time_point fineCq= steady_clock::now();
			duration<float> tempoCq= duration_cast<duration<float>> (fineCq- inizioCq);
			std::cout << "Tempo Comman Queue: " << tempoCq.count() << " seconds" << std::endl;

			
			std::vector<cl::Buffer> buffers;
			// Creo buffers n x 2n per matrice augmentata
			steady_clock::time_point inizioCreazioneBuffer= steady_clock::now();
			std::cout << matrice_augmentata.size() << std::endl;
			buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrice_augmentata.size() * sizeof(cl_double), matrice_augmentata.data(), &operationResult));
			
			// Creo buffer per matrice input
			std::cout << matrice_input.size() << std::endl;
			buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrice_input.size() * sizeof(cl_double), matrice_input.data(), &operationResult));
	
			buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  matrice_augmentata.size() * sizeof(cl_double), matrice_augmentata.data(), &operationResult));

			steady_clock::time_point fineCreazioneBuffer= steady_clock::now();
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING BUFFERS" << std::endl;
				throw operationResult;
			}
			duration<float> tempoCreazioneBuffer = duration_cast<duration<float>> (fineCreazioneBuffer- inizioCreazioneBuffer);
			std::cout << "Tempo Creazione Buffer: " << tempoCreazioneBuffer.count() << " seconds" <<  std::endl;



			// Creo programmi usando i kernel
			steady_clock::time_point inizioCreazioneProgrammi= steady_clock::now();
			cl::Program fix_column_program(context, cl::Program::Sources(1, std::make_pair(fixColumnKernelString.c_str(), fixColumnKernelString.length() + 1)), &operationResult);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING PROGRAM FIX COLUMNS" << std::endl;
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

			cl::Program pivot_kernel_program(context, cl::Program::Sources(1, std::make_pair(pivotKernelString.c_str(), pivotKernelString.length() + 1)), &operationResult);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING PROGRAM PIVOT KERNEL" << std::endl;
				throw operationResult;
			}

			steady_clock::time_point fineCreazioneProgrammi = steady_clock::now();
			duration<float> tempoCreazioneProgrammi = duration_cast<duration<float>> (fineCreazioneProgrammi- inizioCreazioneProgrammi);
			std::cout << "Tempo Creazione Programmi: " << tempoCreazioneProgrammi.count() << " seconds" <<  std::endl;


			///////////////////////////////////////////////////////////////
			/// Compilo i programmi
			// TODO: capire come poter passare solo un  device  e non  la lista intera
			steady_clock::time_point inizioCompilazioneProgrammi= steady_clock::now();
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

			operationResult = matrix_program.build(devices);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR BUILDING MATRIXPROGRAM" << std::endl;
				throw operationResult;
			}
			if (operationResult == CL_BUILD_PROGRAM_FAILURE) {
				std::string err;
				fix_row_program.getBuildInfo(chosenDevice, CL_PROGRAM_BUILD_LOG, &err);
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

			operationResult = pivot_kernel_program.build(devices);
			if (operationResult == CL_BUILD_PROGRAM_FAILURE) {
				std::string err;
				pivot_kernel_program.getBuildInfo(chosenDevice, CL_PROGRAM_BUILD_LOG, &err);
				std::cout << err;
			}
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR BUILDING PROGRAM PIVOT KERNEL" << std::endl;
				throw operationResult;
			
			}

			steady_clock::time_point fineCompilazioneProgrammi = steady_clock::now();
			duration<float> tempoCompilazioneProgrammi = duration_cast<duration<float>> (fineCompilazioneProgrammi - inizioCompilazioneProgrammi);
			std::cout << "Tempo Compilazione Programmi: " << tempoCompilazioneProgrammi.count() << " seconds" << std::endl;


			///////////////////////////////////////////////////////////////
			/// Creo i kernel
			steady_clock::time_point inizioCreazioneKernel = steady_clock::now();
			cl::Kernel fix_column_kernel(fix_column_program, "fixColumnKernel", &operationResult);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING FIX COLUMN KERNEL" << std::endl;
				throw operationResult;
			}

			auto kernelWorkGroupSize = fix_column_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(chosenDevice);

			std::cout << "KERNEL WORK GROUP SIZE: " << kernelWorkGroupSize << std::endl;



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

			cl::Kernel pivot_kernel(pivot_kernel_program, "pivotElementsKernel", &operationResult);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING PIVOT KERNEL" << std::endl;
				throw operationResult;
			}

			steady_clock::time_point fineCreazioneKernel= steady_clock::now();
			duration<float> tempoCreazioneKernel= duration_cast<duration<float>> (fineCreazioneKernel- inizioCreazioneKernel);
			std::cout << "Tempo Creazione Kernel: " << tempoCreazioneKernel.count() << " seconds" << std::endl;


			/// Imposto argomenti kernel + esecuzione kernel 
			///
			/// Con dei cicli itero attraverso le colonne/righe della matrice augmentata
			/// Ad ogni ciclo imposto nuovi parametri al kernel e procedo con una nuova esecuzione
			// Ci pensa openCL ad aspettare che un  kernel finisca prima di inizaire l'altro
			// MAKE AUGMENTED MATRIX
			operationResult = make_augmented_matrix_kernel.setArg(0, buffers[0]);
			operationResult = make_augmented_matrix_kernel.setArg(1, buffers[1]);
			operationResult = make_augmented_matrix_kernel.setArg(2, matrix_order);
			operationResult = commandQueue.enqueueNDRangeKernel(make_augmented_matrix_kernel, cl::NDRange(0,1), cl::NDRange((cl_int)(matrix_order*2), matrix_order), cl::NullRange, NULL, NULL);
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


			// ROWS
			operationResult = fix_row_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata
			// COLUMNS
			operationResult = fix_column_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata
			// PIVOT 
			operationResult = pivot_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata

			tempoComputazioneInizio = steady_clock::now();
			duration<float> pivotComputeTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
			duration<float> pivotTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
			duration<float> rowTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
			duration<float> columnTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
			duration<float> readWriteTime = duration_cast<duration<float>> (steady_clock::now() - steady_clock::now());
			

			for (int i = 0; i < matrix_order; i++) { 

				steady_clock::time_point pivotMaxInizio = steady_clock::now();
				if ((i%2) == 0) 
					operationResult = commandQueue.enqueueReadBuffer(buffers[0], CL_TRUE, 0, matrice_augmentata.size() * sizeof(cl_double), matrice_augmentata.data(), NULL);
				else 
					operationResult = commandQueue.enqueueReadBuffer(buffers[2], CL_TRUE, 0, matrice_augmentata.size() * sizeof(cl_double), matrice_augmentata.data(), NULL);

				cl_int rigaMax= i;
				cl_double pivotMax = 0.0;
				for (int k = i; k < matrix_order; k++) {
					if (abs(matrice_augmentata[k * matrix_order * 2 +i]) >= abs(matrice_augmentata[rigaMax * matrix_order * 2 + i])) {
						rigaMax = k;
					}
				}
				pivotMax = matrice_augmentata[rigaMax * matrix_order * 2 + i];
				if (pivotMax == 0) {
					std::cout << "PIVOT INVALIDO: " << pivotMax << std::endl;
					throw;
				}
				steady_clock::time_point pivotMaxFine = steady_clock::now();
				pivotComputeTime +=  duration_cast<duration<float>> (pivotMaxFine- pivotMaxInizio);


				// PIVOT
				steady_clock::time_point pivotInizio = steady_clock::now();
				if((i%2) == 0)
					operationResult = pivot_kernel.setArg(0, buffers[0]);
				else 
					operationResult = pivot_kernel.setArg(0, buffers[2]);

				operationResult = pivot_kernel.setArg(2, i); // index riga su cui fare il pivot 
				operationResult = pivot_kernel.setArg(3, rigaMax); 
				operationResult = commandQueue.enqueueNDRangeKernel(pivot_kernel, cl::NullRange, cl::NDRange((cl_int)(matrix_order*2)), cl::NullRange, NULL, NULL);
				steady_clock::time_point pivotFine = steady_clock::now();
				pivotTime +=  duration_cast<duration<float>> (pivotFine- pivotInizio);

				// ROWS
				if ((i % 2) == 0) 
					operationResult = fix_row_kernel.setArg(0, buffers[0]); 
				else 
					operationResult = fix_row_kernel.setArg(0, buffers[2]); 

				steady_clock::time_point rowInizio = steady_clock::now();
				operationResult = fix_row_kernel.setArg(2, i); // index riga da fixare
				operationResult = fix_row_kernel.setArg(3, pivotMax); // index riga da fixare
				operationResult = commandQueue.enqueueNDRangeKernel(fix_row_kernel, cl::NullRange, cl::NDRange(matrix_order*2), cl::NDRange(256), NULL, NULL);
				steady_clock::time_point rowFine = steady_clock::now();
				rowTime +=  duration_cast<duration<float>> (rowFine - rowInizio);

				// COLUMNS
				if ((i % 2) == 0) {
					operationResult = fix_column_kernel.setArg(0, buffers[0]); // read
					operationResult = fix_column_kernel.setArg(3, buffers[2]); // write
				}
				else {
					operationResult = fix_column_kernel.setArg(0, buffers[2]); // read 
					operationResult = fix_column_kernel.setArg(3, buffers[0]); // write
				}
				steady_clock::time_point colInizio = steady_clock::now();
				operationResult = fix_column_kernel.setArg(2, i); // index colonna da fixare
				operationResult = commandQueue.enqueueNDRangeKernel(fix_column_kernel, cl::NullRange, cl::NDRange((cl_int)(matrix_order*2), matrix_order), cl::NullRange, NULL, NULL);
				steady_clock::time_point colFine= steady_clock::now();
				columnTime +=  duration_cast<duration<float>> (colFine - colInizio);
	
/*
				steady_clock::time_point rwInizio = steady_clock::now();
				operationResult = commandQueue.finish();
				operationResult = commandQueue.enqueueReadBuffer(buffers[2], CL_TRUE, 0, matrice_augmentata.size() * sizeof(cl_double), matrice_augmentata.data(), NULL);
				operationResult = commandQueue.finish();
				operationResult = commandQueue.enqueueWriteBuffer(buffers[0], CL_TRUE, 0, matrice_augmentata.size() * sizeof(cl_double), matrice_augmentata.data(), NULL);
				steady_clock::time_point rwFine = steady_clock::now();
				readWriteTime +=  duration_cast<duration<float>> (rwFine - rwInizio);
*/
				operationResult = commandQueue.finish();
			}
		

			std::cout << "\n\n";
			std::cout << "TEMPO COMPUTE PIVOT: " << pivotComputeTime.count() << " s" << std::endl;
			std::cout << "TEMPO PIVOT: " << pivotTime.count() << " s" << std::endl;
			std::cout << "TEMPO ROW: " << rowTime.count() << " s" << std::endl;
			std::cout << "TEMPO COLUMN: " << columnTime.count() << " s" << std::endl;
			std::cout << "TEMPO READ WRITE: " << readWriteTime.count() << " s" << std::endl;
			std::cout << "\n\n";

			operationResult = commandQueue.finish();
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING DEVICES" << std::endl;
				throw operationResult;
			}


			// TODO: TENERE QUESTO READ BUFFER PERCHé SERVER PER IL CONTROLLO DELLA MATRICE IDENTITA
			if(((matrix_order - 1) % 2) == 0)
				operationResult = commandQueue.enqueueReadBuffer(buffers[2], CL_TRUE, 0, matrice_augmentata.size() * sizeof(cl_double), m.data(), NULL);
			else
				operationResult = commandQueue.enqueueReadBuffer(buffers[0], CL_TRUE, 0, matrice_augmentata.size() * sizeof(cl_double), m.data(), NULL);

			operationResult = commandQueue.finish();
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING DEVICES" << std::endl;
				throw operationResult;
			}

			steady_clock::time_point tempoComputazioneFine = steady_clock::now();
			duration<float> tempoComputazioneGPU = duration_cast<duration<float>> (tempoComputazioneFine - tempoComputazioneInizio);
	
			duration<float> tot = pivotComputeTime + pivotTime + rowTime + columnTime + readWriteTime;
			std::cout << "ERRORE TIME: " << (tempoComputazioneGPU - tot).count() << std::endl;
			
			// GET INVERTED MATRIX 
			if(((matrix_order - 1) % 2) == 0)
				operationResult = get_inverted_matrix_kernel.setArg(0, buffers[2]);
			else	
				operationResult = get_inverted_matrix_kernel.setArg(0, buffers[0]);
			
			operationResult = get_inverted_matrix_kernel.setArg(1, buffers[1]);
			operationResult = get_inverted_matrix_kernel.setArg(2, matrix_order);

			operationResult = commandQueue.finish();
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING DEVICES" << std::endl;
				throw operationResult;
			}

			operationResult = commandQueue.enqueueNDRangeKernel(get_inverted_matrix_kernel, cl::NDRange(0,1), cl::NDRange((cl_int)(2*matrix_order), matrix_order), cl::NullRange, NULL, NULL);
			if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR GET INVERTED MATRIX KERNEL EXECUTION" << std::endl;
					throw operationResult;
			}

			operationResult = commandQueue.finish();
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING DEVICES" << std::endl;
				throw operationResult;
			}


			// BANDWIDTH
			// Theoretical max: (1780MHz * (256/8)*2 bit)/1e9 =  113 GB/s
			duration<float> tempoFixRow = duration_cast<duration<float>> (tempoFixRowFine - tempoFixRowInizio);
			std::cout << "Bandwidth: " << ((matrix_order*matrix_order*2*2)/readWriteTime.count())/1e9 << " GB/s" << std::endl;


			std::vector<cl_double> matriceResult= std::vector<cl_double>(matrice_input.size(), 0.0);
			steady_clock::time_point inizioRead = steady_clock::now();
			// NB: la matrice augmentata � il doppio rispetto al numero di elementi iniziali
			operationResult = commandQueue.enqueueReadBuffer(buffers[1], CL_TRUE, 0, matriceResult.size() * sizeof(cl_double), matriceResult.data(), NULL);

			operationResult = commandQueue.finish();
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING DEVICES" << std::endl;
				throw operationResult;
			}

			steady_clock::time_point fineRead = steady_clock::now();
			duration<float> tempoRead = duration_cast<duration<float>> (fineRead - inizioRead);
			std::cout << "Tempo Read: " << tempoRead.count() << " seconds" << std::endl;
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR ENQUEUE READ BUFFER" << std::endl;
				throw operationResult;
			}

		
			steady_clock::time_point tempoTotaleFine= steady_clock::now();
			duration<float> tempoTotale = duration_cast<duration<float>> (tempoTotaleFine - tempoTotaleInizio);
			std::cout << "Tempo Totale Impiegato: " << tempoTotale.count() << " seconds" <<  std::endl;
			std::cout << "Tempo Computazione: " << tempoComputazioneGPU.count() << " seconds" <<std::endl;
		
			// CONTROLLO CHE MATRICE AUGMENTATA ABBIA MATRICE IDENTITYA A SINISTRA
			int row = 0; 
			for (int i = 0; i < m.size(); i++) {
				if ((row*matrix_order*2 + row) == i) {
					if (m[i] != 1) {
						std::cout << "DIAGONALE DIVERSA DA 1: \t\t" << m[i] << " != 1";
						return {};
					}
				}
				else if (i < (row*matrix_order*2 + matrix_order)){
					if (m[i] != 0) {
						std::cout << "NON DIAGONALE DIVERSA DA 0: \t\t " << m[i] << " != 0";
						return {};
					}
				}
				else {
					i += matrix_order;
					if (i != 0 && (i % (matrix_order * 2) == 0)) {
						row++;
					}
				}
			}

			std::cout << " \n\nNESSUN ERRORE CON CONTROLLO MATRICE IDENTITA DELLA MATRICE AUGMENATA \n\n"; 

			buffers.clear();
			return matriceResult;
		}
		catch (cl_int e) {
			std::cerr << "ERRORE N°: " << e << std::endl;
		}
		return {};
	}




