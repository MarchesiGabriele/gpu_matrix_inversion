#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
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
		std::vector<float> matrice_input = { 1,2,3,4,5,6,7,8,9 };

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
		commandQueue = cl::CommandQueue(chosenDevice);

		// Creo buffers n x 2n
		cl::Buffer augmented_matrix(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrice_augmentata.size() * sizeof(float)*2, matrice_augmentata.data(), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING BUFFERS" << std::endl;
			throw operationResult;
		}
		///////////////////////////////////////////////////////////////
		/// Recupero kernel dal file esterno

		//fix column kernel
		std::ifstream fix_column_kernelFile("fix_column_kernel.cl");
		std::string fix_column_src(std::istreambuf_iterator<char>(fix_column_kernelFile), (std::istreambuf_iterator<char>()));
		//fix row kernel
		std::ifstream fix_row_kernelFile("fix_row_kernel.cl");
		std::string fix_row_src(std::istreambuf_iterator<char>(fix_row_kernelFile), (std::istreambuf_iterator<char>()));

		///////////////////////////////////////////////////////////////
		/// Creo programma usando il kernel

		//fix column program
		cl::Program fix_column_program(context, cl::Program::Sources(1, std::make_pair(fix_column_src.c_str(), fix_column_src.length() + 1)), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING PROGRAM" << std::endl;
			throw operationResult;
		}
		//fix row program
		cl::Program fix_row_program(context, cl::Program::Sources(1, std::make_pair(fix_row_src.c_str(), fix_row_src.length() + 1)), &operationResult);
		if (operationResult != CL_SUCCESS) {
			std::cerr << "ERROR CREATING PROGRAM" << std::endl;
			throw operationResult;
		}

		// esecuzione kernel per ogni riga della matrice augmentata
		for (int i = 0; i < matrix_order; i++) {
			
		}


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