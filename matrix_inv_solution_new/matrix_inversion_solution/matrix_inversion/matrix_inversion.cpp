#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "matrix_inversion.h"
#define __CL_ENABLE_EXCEPTIONS




	std::vector<float> matrix_inversion(std::vector<float> matrix_vector, int matrix_order) {

		// KERNEL PER FIXARE COLONNE
		const std::string fixColumnKernelString = R"(__kernel void fixColumnKernel(__global float *matrix, int size, int colId){
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

		const std::string fixRowKernelString = R"(__kernel void fixRowKernel(__global float *matrix, int size, int rowId){

	__local float row[100];

	__local float Aii;

	/* scorro gli elementi della riga */
	int colId = get_global_id(0);

	row[colId] = matrix[size*rowId + colId];
	Aii = matrix[size*rowId + rowId];

	row[colId] = row[colId]/Aii;
	matrix[size*rowId + colId] = row[colId];
	})";

		const std::string pivotKernelString = R"(__kernel void pivotElementsKernel(__global float *matrix, int size, int rowId){

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


		// se altezza vettore � zero ritorno vettore vuoto
		if (matrix_order == 0) {
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

			std::string deviceName;

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

			// TODO: PERMETTERE ALL'UTENTE DI SCEGLIERE QUALE PIATTAFORMA USARE
			chosenPlatform = platforms[0];


			// Recupero i device disponibili
			operationResult = chosenPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING DEVICES" << std::endl;
				throw operationResult;
			}

			std::cout << "Device disponibili: " << std::endl;
			for (cl::Device i : devices) {
				i.getInfo(CL_DEVICE_NAME, &deviceName);
				std::cout << "DEVICE: " << deviceName << std::endl;
				std::cout << std::endl;
			}


			// TODO: PERMETTERE ALL'UTENTE DI SCEGLIERE QUALE DEVICE USARE
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
			/// Recupero kernel dal file esterno
		//	std::ifstream fix_column_kernelFile("fix_column_kernel.cl");
		//	std::string fix_column_src(std::istreambuf_iterator<char>(fix_column_kernelFile), (std::istreambuf_iterator<char>()));

			//std::ifstream fix_row_kernelFile("fix_row_kernel.cl");
			//std::string fix_row_src(std::istreambuf_iterator<char>(fix_row_kernelFile), (std::istreambuf_iterator<char>()));


			//std::ifstream pivot_kernel_file("pivot_kernel.cl");
			//std::string pivot_kernel_src(std::istreambuf_iterator<char>(pivot_kernel_file), (std::istreambuf_iterator<char>()));


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
			std::cout << "Nomi Kernel fix column: " << nomi_kernel << std::endl;
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING PROGRAM INFO" << std::endl;
				throw operationResult;
			}

			operationResult = fix_row_program.getInfo(CL_PROGRAM_KERNEL_NAMES, &nomi_kernel);
			std::cout << "Nomi Kernel fix row: " << nomi_kernel << std::endl;
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR GETTING PROGRAM INFO" << std::endl;
				throw operationResult;
			}

			operationResult = pivot_kernel_program.getInfo(CL_PROGRAM_KERNEL_NAMES, &nomi_kernel);
			std::cout << "Nomi Kernel pivot: " << nomi_kernel << std::endl;
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
			/// Prima eseguo il fix_row_kernel e poi il fix_column_kernel in modo alternato
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
			for (int i = 0; i < matrix_order; i++) {
				// PIVOT 
				operationResult = pivot_kernel.setArg(2, i); // index riga su cui fare il pivot 
				// come dimensione globale ho usato "2 * matrix_order, matrix_order" perch� se serve fare almeno un pivot, vengono toccati tutti gli elementi della matrice augmentata 
				operationResult = commandQueue.enqueueNDRangeKernel(pivot_kernel, cl::NullRange, cl::NDRange(2 * matrix_order, matrix_order), cl::NullRange, NULL, NULL);
				if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR SETTING ARGUMENT PIVOT KERNEL" << std::endl;
					throw operationResult;
				}

				// ROWS
				operationResult = fix_row_kernel.setArg(2, i); // index riga da fixare
				// come dimensione globale ho usato "2 * matrix_order, 1" perch� ogni kernel esegue l'operazione su tutti gli elementi di una sola riga 
				operationResult = commandQueue.enqueueNDRangeKernel(fix_row_kernel, cl::NullRange, cl::NDRange(2 * matrix_order, 1), cl::NullRange, NULL, NULL);
				if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR SETTING ARGUMENT FIX ROW KERNEL" << std::endl;
					throw operationResult;
				}

				// COLUMNS
				operationResult = fix_column_kernel.setArg(2, i); // index colonna da fixare
				// come dimensione globale ho usato "2 * matrix_order, matrix_order" perch� ogni kernel esegue l'operazione su tutta la matrice augmentata
				// ogni kernel considera una colonna da sistemare, ma per ogni elemento della colonna devo fixare l'intera riga quindi eseguo operazioni su tutti gli elementi della matrice augmentata
				operationResult = commandQueue.enqueueNDRangeKernel(fix_column_kernel, cl::NullRange, cl::NDRange(2 * matrix_order, matrix_order), cl::NullRange, NULL, NULL);
				if (operationResult != CL_SUCCESS) {
					std::cerr << "ERROR ROW KERNEL EXECUTION" << std::endl;
					throw operationResult;
				}
			}
			operationResult = commandQueue.finish();

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


			return result;
		}
		catch (cl_int e) {
			std::cerr << "ERRORE N�: " << e << std::endl;
		}
		return {};
	}



	int main(){
		std::vector<float> vet = {1,3,5,2,4,3,2,3,4};
		int ordine = sqrt(vet.size());
		std::vector<float> c =  matrix_inversion(vet, ordine);

		for (int i = 0; i < c.size(); i++) {
			if ( i % ordine == 0) {
				std::cout <<std::endl;
			}
			std::cout << c[i] << "\t\t";
		}

	}

