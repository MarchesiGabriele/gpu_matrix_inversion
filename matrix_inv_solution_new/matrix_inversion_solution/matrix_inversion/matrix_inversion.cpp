#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#define __CL_ENABLE_EXCEPTIONS


	std::vector<float> matrix_inversion(std::vector<float> matrix_vector, int matrix_order) {
		// KERNEL PER FIXARE COLONNE
		const std::string fixColumnKernelString = R"(__kernel void fixColumnKernel(__global float *matrix, int size, int colId){
		
		/* valori di una riga, itero matrice orizzontalmente*/
		int j = get_global_id(0);
	
		/* valori di una colonna, itero matrice verticalmente*/
		int i = get_global_id(1);
	
		/* colonna indicata da colId, formata da "size" elementi */ 
		__local float col[1];	

		/* j-esimo elemento della riga corrispondente a colId */
		__local float AColIdj;

		/* colonna indicata da i, formata da "size" elementi*/
		__local float colj[1];

		col[i] = matrix[i*size+ colId];

		/* controllo se elemento è diverso da zero, se lo è già non devo fare nulla*/
		/* controllo anche di non essere sulla diagonale */
		if(col[i] != 0 || i != colId){
			colj[i] = matrix[i*size+j];
			AColIdj = matrix[colId*size + j];

			colj[i] = colj[i] - AColIdj * col[i];
			matrix[i*size + j] = colj[i];
		}
		})";

		const std::string fixRowKernelString = R"(__kernel void fixRowKernel(__global float *matrix, int size, int rowId){

		__local float row[1];

		__local float Aii;

		/* scorro orizzontalmente la matrice */
		int colId = get_global_id(0);
			Aii = matrix[size*rowId + rowId];
		if(Aii == 1){
			return;
		}

		row[colId] = matrix[size*rowId + colId];

		row[colId] = row[colId]/Aii;
		matrix[size*rowId + colId] = row[colId];
		})";

		// Eseguo pivoting per evitare che elementi della diagonale siano = 0. 
		// Se trovo un elemento sulla diagonale = 0, prendo le altre righe e gliele sommo. 
		const std::string pivotKernelString = R"(__kernel void pivotElementsKernel(__global float *matrix, int size, int rowId){

		__local float selectedRow[0];

		__local float Aii;

		/* itero matrice orizzontalmente*/
		int col = get_global_id(0);
	
		/* itero matrice verticalmente*/
		int row = get_global_id(1);
			
		/* elemento sulla diagonale della matrice */
		Aii = matrix[size*rowId + rowId];

		if(Aii == 0){
			/* riempio la riga corrispondente al mio rowId */
			selectedRow[col] = matrix[size*rowId + col];

			for(int i = 0; i<size; i++){
				/* evito di sommare la stessa riga a se stessa */
				if(rowId != row){
					selectedRow[col] = selectedRow[col] + matrix[size*row + col];
				}
			}
			matrix[size*rowId + col] = selectedRow[col];
		}
		})";


		// se altezza vettore � zero ritorno vettore vuoto
		if (matrix_order <= 0) {
			return {};
		}

		// Controllo se la matrice � quadrata, se non lo � ritorno vettore vuoto
		double matrix_height = matrix_vector.size() / matrix_order * 1.0;
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


			// DEVICE INFO
			std::string deviceName;
			cl_ulong globalCacheSize;
			cl_ulong localMemSize;
			cl_ulong maxConstBufferSize;
			size_t maxWorkGroupSize;
			cl_uint maxWorkItemDimensions;
			cl_ulong globalMemSize;

			// primo parametro funzione -> matrice da invertire (sofforma di vettore o vettore di vettori)
			std::vector<float> matrice_input = matrix_vector;

			// Ordine Matrice  
			// TODO CONTROLLARE  CHE LA MATRICE INSERITA SIA  QUADRATA !!
			int matrix_order = sqrt(matrice_input.size());

			// matrice identita con stessa dimensione della matrice passata come input
			std::vector<float> matrice_indentita(matrice_input.size(), 0);

			// matrice input a sinistra e matrice identita a destra (n x 2n)
			// la dimensione di questa matrice � la dimensione globale
			std::vector<float> matrice_augmentata = {};

				
			using namespace std::chrono;
			// Tempo totale impiegato per eseguire la funzione matrix_inversio() 
			steady_clock::time_point tempoTotaleInizio = steady_clock::now();
			// Tempo impiegato dalla GPU per eseguire i kernel
			steady_clock::time_point tempoComputazioneInizio;


			// creo matrice identita
			int index = 0;
			for (int i = 0; i < matrix_order; i++) {
				matrice_indentita[index] = 1;
				index += matrix_order + 1;
			}

			// creo matrice augmentata
			int colonna = 0;
			int indexIdentita = 0;
			int indexInput = 0;
			for (int i = 0; i < matrice_input.size() * 2; i++) {
				// inserisco matrice identita
				if (colonna >= matrix_order) {
					matrice_augmentata.push_back(matrice_indentita[indexIdentita]);
					indexIdentita++;
				}
				// inserisco matrice input
				else {
					matrice_augmentata.push_back(matrice_input[indexInput]);
					indexInput++;
				}
				colonna++;
				if (std::fmod(colonna, 2 * matrix_order) == 0) {
					colonna = 0;
				}
			}


			// Recupero le piattaforme disponibili
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

			// Scelgo piattaforma AMD 
			chosenPlatform = platforms[0];

			// Recupero i device disponibili
			operationResult = chosenPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING DEVICES" << std::endl;
				throw operationResult;
			}

			// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html
			std::cout << "Device disponibili: " << std::endl;
			for (cl::Device i : devices) {
				i.getInfo(CL_DEVICE_NAME, &deviceName);
				i.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &globalCacheSize);
				i.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
				i.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemSize);
				i.getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &maxConstBufferSize);
				i.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
				i.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &maxWorkItemDimensions);
				std::cout << "DEVICE: " << deviceName << std::endl;
				std::cout << "Global Cache size: " << globalCacheSize << " bytes" << std::endl;
				std::cout << "Global mem size: " << globalMemSize  << " bytes" << std::endl;
				std::cout << "Local mem size: " <<localMemSize << " bytes"<< std::endl;
				std::cout << "Max Const Buffer Size: " << maxConstBufferSize << " bytes"<< std::endl;
				std::cout << "Max Work Group Size: " << maxWorkGroupSize << std::endl;
				std::cout << "Max Work Item dimensions: " << maxWorkItemDimensions << std::endl;
				std::cout << std::endl;
			}

			// Scelgo scheda grafica come DEVICE
			chosenDevice = devices[0];

			// Creo il context per il device scelto
			context = cl::Context(chosenDevice);

			// Creo la command queue per il device scelto
			commandQueue = cl::CommandQueue(context, chosenDevice);

			// Creo buffers n x 2n
			std::cout << matrice_augmentata.size() << std::endl;
			cl::Buffer augmented_matrix(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrice_augmentata.size() * sizeof(float), matrice_augmentata.data(), &operationResult);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING BUFFERS" << std::endl;
				throw operationResult;
			}


			///////////////////////////////////////////////////////////////
			/// Creo programma usando il kernel
			cl::Program fix_column_program(context, cl::Program::Sources(1, std::make_pair(fixColumnKernelString.c_str(), fixColumnKernelString.length() + 1)), &operationResult);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING PROGRAM FIX COLUMNS" << std::endl;
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



			///////////////////////////////////////////////////////////////
			/// Compilo i programmi
			// TODO: capire come poter passare solo un  device  e non  la lista intera
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
				std::cerr << "ERROR BUILDING PROGRAM pIVOT KERNEL" << std::endl;
				throw operationResult;
			}


			// RECUPERO INFO SUI PROGRAMMI
			std::string nomi_kernel;
			operationResult = fix_column_program.getInfo(CL_PROGRAM_KERNEL_NAMES, &nomi_kernel);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING PROGRAM INFO" << std::endl;
				throw operationResult;
			}

			operationResult = fix_row_program.getInfo(CL_PROGRAM_KERNEL_NAMES, &nomi_kernel);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING PROGRAM INFO" << std::endl;
				throw operationResult;
			}

			operationResult = pivot_kernel_program.getInfo(CL_PROGRAM_KERNEL_NAMES, &nomi_kernel);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING PROGRAM INFO" << std::endl;
				throw operationResult;
			}



			///////////////////////////////////////////////////////////////
			/// Creo i kernel
			cl::Kernel fix_column_kernel(fix_column_program, "fixColumnKernel", &operationResult);

			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING FIX COLUMN KERNEL" << std::endl;
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

			///////////////////////////////////////////////////////////////
			/// Imposto argomenti kernel + esecuzione kernel 
			///
			/// Con dei cicli itero attraverso le colonne/righe della matrice augmentata
			/// Ad ogni ciclo imposto nuovi parametri al kernel e procedo con una nuova esecuzione
			// Ci pensa openCL ad aspettare che un  kernel finisca prima di inizaire l'altro

			// ROWS
			operationResult = fix_row_kernel.setArg(0, augmented_matrix);
			operationResult = fix_row_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata
			// COLUMNS
			operationResult = fix_column_kernel.setArg(0, augmented_matrix);
			operationResult = fix_column_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata
			// PIVOT 
			operationResult = pivot_kernel.setArg(0, augmented_matrix);
			operationResult = pivot_kernel.setArg(1, matrix_order * 2); // larghezza matrice augmentata
			tempoComputazioneInizio = steady_clock::now();
			for (int i = 0; i < matrix_order; i++) { 
				// PIVOT
				operationResult = pivot_kernel.setArg(2, i); // index riga su cui fare il pivot 
				// come dimensione globale ho usato "2 * matrix_order, matrix_order" perch� se serve fare almeno un pivot, vengono toccati tutti gli elementi della matrice augmentata 
				operationResult = commandQueue.enqueueNDRangeKernel(pivot_kernel, cl::NullRange, cl::NDRange(matrix_order, matrix_order), cl::NullRange, NULL, NULL);
				if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR SETTING ARGUMENT PIVOT KERNEL" << std::endl;
					throw operationResult;
				}

				// ROWS
				operationResult = fix_row_kernel.setArg(2, i); // index riga da fixare
				// come dimensione globale ho usato "2 * matrix_order, 1" perch� ogni kernel esegue l'operazione su tutti gli elementi di una sola riga 
				operationResult = commandQueue.enqueueNDRangeKernel(fix_row_kernel, cl::NullRange, cl::NDRange( matrix_order, 1),  cl::NullRange, NULL, NULL);
				if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR SETTING ARGUMENT FIX ROW KERNEL" << std::endl;
					throw operationResult;
				}

				// COLUMNS
				operationResult = fix_column_kernel.setArg(2, i); // index colonna da fixare
				// come dimensione globale ho usato "2 * matrix_order, matrix_order" perch� ogni kernel esegue l'operazione su tutta la matrice augmentata
				// ogni kernel considera una colonna da sistemare, ma per ogni elemento della colonna devo fixare l'intera riga quindi eseguo operazioni su tutti gli elementi della matrice augmentata
				operationResult = commandQueue.enqueueNDRangeKernel(fix_column_kernel, cl::NullRange, cl::NDRange( matrix_order, matrix_order), cl::NullRange, NULL, NULL);
				if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR ROW KERNEL EXECUTION" << std::endl;
					throw operationResult;
				}
			}
			operationResult = commandQueue.finish();
			steady_clock::time_point tempoComputazioneFine = steady_clock::now();
			duration<double> tempoComputazioneGPU = duration_cast<duration<double>> (tempoComputazioneFine - tempoComputazioneInizio);

			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR ENQUEUE READ BUFFER" << std::endl;
				throw operationResult;
			}

			// NB: la matrice augmentata � il doppio rispetto al numero di elementi iniziali
			operationResult = commandQueue.enqueueReadBuffer(augmented_matrix, CL_TRUE, 0, matrice_augmentata.size() * sizeof(float), matrice_augmentata.data(), NULL);

			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR ENQUEUE READ BUFFER" << std::endl;
				throw operationResult;
			}
	

			
			// Recupero solo la matrice inversa da quella augmentata
			std::vector<float> result = {};
			int riga = 0;
			for (int i = 0; i < matrice_augmentata.size(); i++) {
				if ((i %(matrix_order * 2)) == 0) {
					riga++;
				}
				if (i>=((riga-1)*matrix_order*2 + matrix_order)) {
					result.push_back(matrice_augmentata[i]);
				}
			}
		
			// stampo tempo impiegato

			steady_clock::time_point tempoTotaleFine= steady_clock::now();
			duration<double> tempoTotale = duration_cast<duration<double>> (tempoTotaleFine - tempoTotaleInizio);
			std::cout << "Tempo Totale Impiegato: " << tempoTotale.count() << " seconds" <<  std::endl;
			std::cout << "Tempo Computazione: " << tempoComputazioneGPU.count() << " seconds" <<std::endl;
			
			
			return result;
		}
		catch (cl_int e) {
			std::cerr << "ERRORE N�: " << e << std::endl;
		}
		return {};
	}




