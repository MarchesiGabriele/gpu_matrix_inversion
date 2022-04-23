#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#define __CL_ENABLE_EXCEPTIONS



std::vector<int> matrix_inversion(std::vector<int> matrix_vector, int matrix_order) {

	// se altezza vettore è zero ritorno vettore vuoto
	if (matrix_order == 0){
		return {};
	}

	// Controllo se la matrice è quadrata, se non lo è ritorno vettore vuoto
	double matrix_height= matrix_vector.size() / matrix_order * 1.0;
	if (matrix_height != matrix_order) {
		return {};
	}

	try {
		// lista delle piattaforme disponibili 
		std::vector<cl::Platform>  platforms;
	
		// lista dei device disponibili  
		std::vector<cl::Device>  devices;
	
		// contesto della piattaforma in cui mi trovo
		cl::Context context;

		// Command queue per il kernel ______
		cl::CommandQueue commandQueue;
	
		// Variabile usata per immagazzinare i risultati/errori delle operazioni che vengono eseguite
		cl_int operationResult;



		// Recupero le piattaforme disponibili
		operationResult = cl::Platform::get(&platforms);
		if ( operationResult != CL_SUCCESS) {
			std::cerr << "ERROR GETTING PLATFORM" << std::endl;
			throw operationResult ;
		}
		std::cout << "Piattaforme disponibili: " << platforms.data() << std::endl;


		// Recupero i device disponibili
		


	
	
	
	}
	catch (cl_int e) {
		std::cerr << "ERRORE N°: " << e << std::endl;
	}


	return {};
}



