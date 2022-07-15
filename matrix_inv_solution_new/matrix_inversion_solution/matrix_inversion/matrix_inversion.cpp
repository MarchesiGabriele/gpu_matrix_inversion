#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#define __CL_ENABLE_EXCEPTIONS


	std::vector<float> matrix_inversion(std::vector<float> matrix_vector, int matrix_order) {
		// KERNEL PER FIXARE COLONNE
		const std::string fixColumnKernelString = R"(
		__kernel void fixColumnKernel(__global float *matrix, int size, int colId){
			/* valori colonna matrice*/
			int i = get_global_id(1);
			/* valori riga matrice*/
			int j = get_global_id(0);
			/* colonna indicata da colId */ 
			__local float col[100];	
			/* elemento della riga corrispondente a colId */
			__local float AColIdj;
			/* riga indicata da i */
			__local float colj[100];
			col[i] = matrix[i*size+ colId];
			/* controllo se elemento � diverso da zero, se lo � gi� non devo fare nulla*/
			if(col[i] != 0){
				colj[i] = matrix[i*size+j];
				AColIdj = matrix[colId*size + j];
				/* controllo  di non essere sulla diagonale */
				if(i != colId){
					colj[i] = colj[i] - AColIdj * col[i];
				}
				matrix[i*size + j] = colj[i];
			}
		})";

		const std::string fixRowKernelString = R"(
		__kernel void fixRowKernel(__global float *matrix, int size, int rowId){
			__local float row[100];
			__local float Aii;
			/* scorro gli elementi della riga */
			int colId = get_global_id(0);
			row[colId] = matrix[size*rowId + colId];
			Aii = matrix[size*rowId + rowId];
			row[colId] = row[colId]/Aii;
			matrix[size*rowId + colId] = row[colId];
		})";

		// Eseguo pivoting per evitare che elementi della diagonale siano = 0. 
		// Se trovo un elemento sulla diagonale = 0, prendo le altre righe e gliele sommo. 
		const std::string pivotKernelString = R"(
		__kernel void pivotElementsKernel(__global float *matrix, int size, int rowId){
			__local float selectedRow[100];
			__local float Aii;
			/* itero colonne per ciascuna riga*/
			int col = get_global_id(0);
			/* itero righe */
			int row = get_global_id(1);
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

		// Con matrixOrder si indende l'ordine della matrice iniziale, non la larghezza della matrice augmentata!!
		// Finche sono a sinistra della matrice augmentata, leggo e scrivo dalla matrice input alla matrice augmentata
		// Se sono a destra allora scrivo solamente il valore 1 o 0. 
		// Metto dimensioni globali (2*matrixOrder, matrixOrder)
		// TODO: RICORDO DI METTERE OFFSET DI 1 NELL NDENQUEUERANGEKERNEL!!!!
		const std::string matrixKernelString= R"(
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
		} 
	
		/* La matrice inversa la vado a mettere dentro il buffer della matrice di input. Per evitare di creare un nuovo buffer*/
		__kernel void getInvertedMatrix(__global float *matrix, __global float *inputMatrix, int matrixOrder){
			/* COL VA DA 0 A 2*MATRIX_ORDER */
			int col = get_global_id(0);

			/* ROW VA DA 1 A MATRIX_ORDER */
			int row = get_global_id(1);

			if(col >= matrixOrder){
				inputMatrix[(col-matrixOrder) + (row-1)*matrixOrder] = matrix[col + (row-1)*matrixOrder*2]; 
			}
		}
		)";


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

			// matrice input a sinistra e matrice identita a destra (n x 2n)
			// la dimensione di questa matrice � la dimensione globale
			std::vector<float> matrice_augmentata = std::vector<float>(matrix_order*matrix_order*2, 0);

				
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
			duration<double> tempoRecuperoPlat= duration_cast<duration<double>> (fineRecuperoPlat- inizioRecuperoPlat);
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
			steady_clock::time_point inizioCtx= steady_clock::now();
			context = cl::Context(chosenDevice);
			steady_clock::time_point fineCtx= steady_clock::now();
			duration<double> tempoCtx= duration_cast<duration<double>> (fineCtx- inizioCtx);
			std::cout << "Tempo Context" << tempoCtx.count() << " seconds" << std::endl;

			// Creo la command queue per il device scelto
			steady_clock::time_point inizioCq= steady_clock::now();
			commandQueue = cl::CommandQueue(context, chosenDevice);			
			steady_clock::time_point fineCq= steady_clock::now();
			duration<double> tempoCq= duration_cast<duration<double>> (fineCq- inizioCq);
			std::cout << "Tempo Comman Queue: " << tempoCq.count() << " seconds" << std::endl;


			// Creo buffers n x 2n per matrice augmentata
			steady_clock::time_point inizioCreazioneBuffer= steady_clock::now();
			std::cout << matrice_augmentata.size() << std::endl;
			cl::Buffer augmented_matrix(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrice_augmentata.size() * sizeof(float), matrice_augmentata.data(), &operationResult);
			
			// Creo buffer per matrice input
			std::cout << matrice_input.size() << std::endl;
			cl::Buffer input_matrix(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrice_input.size() * sizeof(float), matrice_input.data(), &operationResult);
	
			steady_clock::time_point fineCreazioneBuffer= steady_clock::now();
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING BUFFERS" << std::endl;
				throw operationResult;
			}
			duration<double> tempoCreazioneBuffer = duration_cast<duration<double>> (fineCreazioneBuffer- inizioCreazioneBuffer);
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
			duration<double> tempoCreazioneProgrammi = duration_cast<duration<double>> (fineCreazioneProgrammi- inizioCreazioneProgrammi);
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
			duration<double> tempoCompilazioneProgrammi = duration_cast<duration<double>> (fineCompilazioneProgrammi - inizioCompilazioneProgrammi);
			std::cout << "Tempo Compilazione Programmi: " << tempoCompilazioneProgrammi.count() << " seconds" << std::endl;


			// RECUPERO INFO SUI PROGRAMMI
			// TODO: questo penso si possa rimuovere a fine development. Tanto i kernel non dovrebbero contenere errori
			std::string nomi_kernel;
			operationResult = fix_column_program.getInfo(CL_PROGRAM_KERNEL_NAMES, &nomi_kernel);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING PROGRAM INFO" << std::endl;
				throw operationResult;
			}

			operationResult = matrix_program.getInfo(CL_PROGRAM_KERNEL_NAMES, &nomi_kernel);
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
			steady_clock::time_point inizioCreazioneKernel = steady_clock::now();
			cl::Kernel fix_column_kernel(fix_column_program, "fixColumnKernel", &operationResult);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING FIX COLUMN KERNEL" << std::endl;
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

			cl::Kernel pivot_kernel(pivot_kernel_program, "pivotElementsKernel", &operationResult);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR CREATING PIVOT KERNEL" << std::endl;
				throw operationResult;
			}

			steady_clock::time_point fineCreazioneKernel= steady_clock::now();
			duration<double> tempoCreazioneKernel= duration_cast<duration<double>> (fineCreazioneKernel- inizioCreazioneKernel);
			std::cout << "Tempo Creazione Kernel: " << tempoCreazioneKernel.count() << " seconds" << std::endl;


			/// Imposto argomenti kernel + esecuzione kernel 
			///
			/// Con dei cicli itero attraverso le colonne/righe della matrice augmentata
			/// Ad ogni ciclo imposto nuovi parametri al kernel e procedo con una nuova esecuzione
			// Ci pensa openCL ad aspettare che un  kernel finisca prima di inizaire l'altro
			// MAKE AUGMENTED MATRIX
			operationResult = make_augmented_matrix_kernel.setArg(0, augmented_matrix);
			operationResult = make_augmented_matrix_kernel.setArg(1, input_matrix);
			operationResult = make_augmented_matrix_kernel.setArg(2, matrix_order);
			operationResult = commandQueue.enqueueNDRangeKernel(make_augmented_matrix_kernel, cl::NDRange(0,1), cl::NDRange(2*matrix_order, matrix_order), cl::NullRange, NULL, NULL);
			if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR MAKE AUGMENTED KERNEL EXECUTION" << std::endl;
					throw operationResult;
			}
		
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
				operationResult = commandQueue.enqueueNDRangeKernel(pivot_kernel, cl::NullRange, cl::NDRange(2*matrix_order, matrix_order), cl::NullRange, NULL, NULL);
				if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR SETTING ARGUMENT PIVOT KERNEL" << std::endl;
					throw operationResult;
				}

				// ROWS
				operationResult = fix_row_kernel.setArg(2, i); // index riga da fixare
				// come dimensione globale ho usato "2 * matrix_order, 1" perch� ogni kernel esegue l'operazione su tutti gli elementi di una sola riga 
				tempoFixRowInizio = steady_clock::now();
				operationResult = commandQueue.enqueueNDRangeKernel(fix_row_kernel, cl::NullRange, cl::NDRange(2*matrix_order, 1),  cl::NullRange, NULL, NULL);
				tempoFixRowFine = steady_clock::now();
				if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR SETTING ARGUMENT FIX ROW KERNEL" << std::endl;
					throw operationResult;
				}

				// COLUMNS
				operationResult = fix_column_kernel.setArg(2, i); // index colonna da fixare
				// come dimensione globale ho usato "2 * matrix_order, matrix_order" perch� ogni kernel esegue l'operazione su tutta la matrice augmentata
				// ogni kernel considera una colonna da sistemare, ma per ogni elemento della colonna devo fixare l'intera riga quindi eseguo operazioni su tutti gli elementi della matrice augmentata
				operationResult = commandQueue.enqueueNDRangeKernel(fix_column_kernel, cl::NullRange, cl::NDRange(2*matrix_order, matrix_order), cl::NullRange, NULL, NULL);
				if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR ROW KERNEL EXECUTION" << std::endl;
					throw operationResult;
				}
			}

			operationResult = commandQueue.finish();
			steady_clock::time_point tempoComputazioneFine = steady_clock::now();
			duration<double> tempoComputazioneGPU = duration_cast<duration<double>> (tempoComputazioneFine - tempoComputazioneInizio);
		
			// GET INVERTED MATRIX 
			operationResult = get_inverted_matrix_kernel.setArg(0, augmented_matrix);
			operationResult = get_inverted_matrix_kernel.setArg(1, input_matrix);
			operationResult = get_inverted_matrix_kernel.setArg(2, matrix_order);
			operationResult = commandQueue.enqueueNDRangeKernel(get_inverted_matrix_kernel, cl::NDRange(0,1), cl::NDRange(2*matrix_order, matrix_order), cl::NullRange, NULL, NULL);
			if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR GET INVERTED MATRIX KERNEL EXECUTION" << std::endl;
					throw operationResult;
			}

			operationResult = commandQueue.finish();


			// BANDWIDTH
			// Theoretical max: (1780MHz * (256/8)*2 bit)/1e9 =  113 GB/s
			duration<double> tempoFixRow = duration_cast<duration<double>> (tempoFixRowFine - tempoFixRowInizio);
			std::cout << "Bandwidth: " << ((matrix_order*2*4)/tempoFixRow.count())/1e9 << " GB/s" << std::endl;


			std::vector<float> matriceResult= std::vector<float>(matrice_input.size(), 0);
			steady_clock::time_point inizioRead = steady_clock::now();
			// NB: la matrice augmentata � il doppio rispetto al numero di elementi iniziali
			operationResult = commandQueue.enqueueReadBuffer(input_matrix, CL_TRUE, 0, matriceResult.size() * sizeof(float), matriceResult.data(), NULL);
			steady_clock::time_point fineRead = steady_clock::now();
			duration<double> tempoRead = duration_cast<duration<double>> (fineRead - inizioRead);
			std::cout << "Tempo Read: " << tempoRead.count() << " seconds" << std::endl;
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR ENQUEUE READ BUFFER" << std::endl;
				throw operationResult;
			}

		
			steady_clock::time_point tempoTotaleFine= steady_clock::now();
			duration<double> tempoTotale = duration_cast<duration<double>> (tempoTotaleFine - tempoTotaleInizio);
			std::cout << "Tempo Totale Impiegato: " << tempoTotale.count() << " seconds" <<  std::endl;
			std::cout << "Tempo Computazione: " << tempoComputazioneGPU.count() << " seconds" <<std::endl;
			
			
			return matriceResult;
		}
		catch (cl_int e) {
			std::cerr << "ERRORE N°: " << e << std::endl;
		}
		return {};
	}




