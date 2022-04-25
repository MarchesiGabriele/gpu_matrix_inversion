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
		std::vector<float> matrice_iniziale = {};

		// Ordine Matrice  
		// TODO CONTROLLARE  CHE LA MATRICE INSERITA SIA  QUADRATA !!
		int matrix_order = sqrt(matrice_iniziale.size());

		



		



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
		cl::Buffer augmented_matrix(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, )







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