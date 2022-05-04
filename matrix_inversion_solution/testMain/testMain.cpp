#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
#define __CL_ENABLE_EXCEPTIONS



std::vector<int> matrix_inversion(std::vector<int> matrix_vector, int matrix_order) {

	// se altezza vettore è zero ritorno vettore vuoto
	if (matrix_order == 0) {
		return {};
	}

	// Controllo se la matrice è quadrata, se non lo è ritorno vettore vuoto
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
		int x = pow(3,2);
		std::vector<float> matrice_input(x, 1);


		for (int i = 0; i < x; i++) {
			matrice_input[i] = rand();
		}

		std::cout << "matrice piena" << x << std::endl;

		// Ordine Matrice  
		// TODO CONTROLLARE  CHE LA MATRICE INSERITA SIA  QUADRATA !!
		int matrix_order = sqrt(matrice_input.size());

		// matrice identita con stessa dimensione della matrice passata come input
		std::vector<float> matrice_indentita(matrice_input.size(), 0);

		// matrice input a sinistra e matrice identita a destra (n x 2n)
		// la dimensione di questa matrice è la dimensione globale
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
			if (std::fmod(colonna, 2*matrix_order)  == 0 ) {
				colonna = 0;
			}
		}
	

		// Recupero le piattaforme disponibili
		operationResult = cl::Platform::get(&platforms);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GETTING PLATFORM" << std::endl;
			throw operationResult;
		}

		std::cout << "Piattaforme disponibili: "<< std::endl;
		for (cl::Platform i: platforms){ 
			i.getInfo(CL_PLATFORM_NAME, &platformName);
			i.getInfo(CL_PLATFORM_VENDOR, &platformVendor);
			i.getInfo(CL_PLATFORM_VERSION, &platformVersion);

			std::cout << "PIATTAFORMA: " << platformName << std::endl;
			std::cout << "VENDITORE: " << platformVendor<< std::endl;
			std::cout << "VERSIONE: " << platformVersion<< std::endl;

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

		std::cout << "Device disponibili: "<< std::endl;
		for (cl::Device i: devices){ 
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
		std::cout<< matrice_augmentata.size()  << std::endl;
		cl::Buffer augmented_matrix(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrice_augmentata.size() * sizeof(float), matrice_augmentata.data(), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING BUFFERS" << std::endl;
			throw operationResult;
		}


		///////////////////////////////////////////////////////////////
		/// Recupero kernel dal file esterno
		std::ifstream fix_column_kernelFile("fix_column_kernel.cl");
		std::string fix_column_src(std::istreambuf_iterator<char>(fix_column_kernelFile), (std::istreambuf_iterator<char>()));

		std::ifstream fix_row_kernelFile("fix_row_kernel.cl");
		std::string fix_row_src(std::istreambuf_iterator<char>(fix_row_kernelFile), (std::istreambuf_iterator<char>()));


		std::ifstream pivot_kernel_file("pivot_kernel.cl");
		std::string pivot_kernel_src(std::istreambuf_iterator<char>(pivot_kernel_file), (std::istreambuf_iterator<char>()));

		///////////////////////////////////////////////////////////////
		/// Creo programma usando il kernel
		cl::Program fix_column_program(context, cl::Program::Sources(1, std::make_pair(fix_column_src.c_str(), fix_column_src.length() + 1)), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING PROGRAM FIX COLUMNS" << std::endl;
			throw operationResult;
		}

		cl::Program fix_row_program(context, cl::Program::Sources(1, std::make_pair(fix_row_src.c_str(), fix_row_src.length() + 1)), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING PROGRAM FIX ROWS" << std::endl;
			throw operationResult;
		}

		cl::Program pivot_kernel_program(context, cl::Program::Sources(1, std::make_pair(pivot_kernel_src.c_str(), pivot_kernel_src.length() + 1)), &operationResult);
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


		// print matrice augmentata
		for (int i = 0; i < matrice_augmentata.size(); i++) {
			if (fmod(i, matrix_order * 2) == 0) {
				std::cout << std::endl;
			}
			std::cout << matrice_augmentata[i] << "\t\t";
		}

	
		std::cout <<  std::endl;
		std::cout <<  std::endl;
		std::cout <<  std::endl;


			
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
			// come dimensione globale ho usato "2 * matrix_order, matrix_order" perchè se serve fare almeno un pivot, vengono toccati tutti gli elementi della matrice augmentata 
			operationResult = commandQueue.enqueueNDRangeKernel(pivot_kernel, cl::NullRange, cl::NDRange(2 * matrix_order, matrix_order), cl::NullRange, NULL, NULL);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR SETTING ARGUMENT PIVOT KERNEL" << std::endl;
				throw operationResult;
			}

			// ROWS
			operationResult = fix_row_kernel.setArg(2, i); // index riga da fixare
			// come dimensione globale ho usato "2 * matrix_order, 1" perchè ogni kernel esegue l'operazione su tutti gli elementi di una sola riga 
			operationResult = commandQueue.enqueueNDRangeKernel(fix_row_kernel, cl::NullRange, cl::NDRange(2 * matrix_order, 1), cl::NullRange, NULL, NULL);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR SETTING ARGUMENT FIX ROW KERNEL" << std::endl;
				throw operationResult;
			}
			
			// COLUMNS
			operationResult = fix_column_kernel.setArg(2, i); // index colonna da fixare
			// come dimensione globale ho usato "2 * matrix_order, matrix_order" perchè ogni kernel esegue l'operazione su tutta la matrice augmentata
			// ogni kernel considera una colonna da sistemare, ma per ogni elemento della colonna devo fixare l'intera riga quindi eseguo operazioni su tutti gli elementi della matrice augmentata
			operationResult = commandQueue.enqueueNDRangeKernel(fix_column_kernel, cl::NullRange, cl::NDRange(2 * matrix_order, matrix_order), cl::NullRange, NULL, NULL);
			if (operationResult != CL_SUCCESS) {
				std::cerr << "ERROR ROW KERNEL EXECUTION" << std::endl;
				throw operationResult;
			}
		}
		std::cout << "bip" << std::endl;
		operationResult = commandQueue.finish();
		std::cout << "bop" << std::endl;

		if (operationResult!= CL_SUCCESS) {
			std::cerr << "ERROR ENQUEUE READ BUFFER" << std::endl;
			throw operationResult;
		}


		std::cout<< matrice_augmentata.size()  << std::endl;
		

		// NB: la matrice augmentata è il doppio rispetto al numero di elementi iniziali
		operationResult = commandQueue.enqueueReadBuffer(augmented_matrix, CL_TRUE, 0, matrice_augmentata.size() * sizeof(float), matrice_augmentata.data(), NULL);

		if (operationResult!= CL_SUCCESS) {
			std::cerr << "ERROR ENQUEUE READ BUFFER" << std::endl;
			throw operationResult;
		}

	
		// PRINT FINAL
		for (int i = 0; i < matrice_augmentata.size(); i++) {
			if (fmod(i, matrix_order * 2) == 0) {
				std::cout << std::endl;
			}
			std::cout << matrice_augmentata[i] << "\t\t";
		}
		std::cout <<  std::endl;
		std::cout <<  std::endl;
		std::cout <<  std::endl;
	}
	catch (cl_int e) {
		std::cerr << "ERRORE N°: " << e << std::endl;
	}
	return {};
}


int main() {
	std::vector<int> result = matrix_inversion({1,2,3,4,5,6,7,8,9}, 3);
	//std::cout << result.data() << std::endl;
	return 0;

}