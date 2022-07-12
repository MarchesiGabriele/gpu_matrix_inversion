#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#define __CL_ENABLE_EXCEPTIONS




// NB: posso moltiplicare solamente matrici quadrate della stessa dimensione
void matrix_multiply(std::vector<float> matriceB, std::vector<float> matriceA) {
	try {
		const std::string kernel = R"(
			__kernel void simpleMultiply(
				__global float* outputC,
				int ordine,
				__global float* inputA,
				__global float* inputB) {
			int row = get_global_id(1);
			int col = get_global_id(0);

			float sum = 0.0f;

			for (int i = 0; i < ordine; i++) {
				sum += inputA[row * ordine+ i] * inputB[i * ordine+ col];
			}

			outputC[row * ordine + col] = sum;

			}
		)"; 



		std::vector<cl::Platform> platform;
		std::vector<cl::Device> device;
		cl::Context context;
		cl::CommandQueue commandQueue;

		std::vector<float> vettoreC(sqrt(matriceA.size())* sqrt(matriceB.size()));
		cl_int result;

		/// SETUP
		// recupero piattaforma
		result = cl::Platform::get(&platform);
		if (result != CL_SUCCESS) {
			std::cerr << "ERROR GETTING PLATFORM" << std::endl;
			throw result;
		}

		// recupero device 
		// TODO: CAPIRE AUTOMATICAMENTE COME PRENDERE LA GPU FISICA E NON QUELLA INTEGRATA DEL PROCESSORE
		result = platform[0].getDevices(CL_DEVICE_TYPE_GPU, &device);
		if (result != CL_SUCCESS) {
			std::cerr << "ERROR GETTING DEVICE" << std::endl;
			throw result;
		}
	

		// il device al context	
		context = cl::Context(device[0]);

		// creo command queue
		commandQueue = cl::CommandQueue(context, device[0]);


		// CREAZIONE BUFFER E MOVIMENTO DATI IN MEMORIA
		// NB: i buffer che contengono delle matrici sono degli array di float!!
		cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matriceA.size() * sizeof(float), matriceA.data(), &result);
		cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matriceB.size() * sizeof(float), matriceB.data(), &result);
		cl::Buffer bufferC(context, CL_MEM_READ_ONLY, vettoreC.size() * sizeof(float), NULL, &result);
		if (result != CL_SUCCESS) {
			std::cerr << "ERROR CREATING BUFFERS" << std::endl;
			throw result;
		}


		// creo programma usando il kernel
		cl::Program program(context, cl::Program::Sources(1, std::make_pair(kernel.c_str(), kernel.length() + 1)), &result);

		if (result != CL_SUCCESS) {
			std::cerr << "ERROR CREATING PROGRAM" << std::endl;
			throw result;
		}


		// compilo il programma
		result = program.build(device);
		if (result != CL_SUCCESS) {
			std::cerr << "ERROR BUILDING PROGRAM" << std::endl;
			throw result;
		}


		// creo kernel
		cl::Kernel kernelMultiply(program, "simpleMultiply", &result);

		if (result != CL_SUCCESS) {
			std::cerr << "ERROR CREATING KERNEL" << std::endl;
			throw result;
		}


		// imposto gli argomenti del kernel	
		result = kernelMultiply.setArg(0, bufferC);
		result = kernelMultiply.setArg(1, (int)sqrt(matriceA.size()));
		result = kernelMultiply.setArg(2, bufferA);
		result = kernelMultiply.setArg(3, bufferB);

		if (result != CL_SUCCESS) {
			std::cerr << "ERROR SETTING ARGUMENTS" << std::endl;
			throw result;
		}



		// eseguo il kernel
		//NB: NDrange sono le dimensioni globali e locali
		result = commandQueue.enqueueNDRangeKernel(kernelMultiply, cl::NullRange, cl::NDRange((int)sqrt(matriceA.size()), (int)sqrt(matriceA.size())), cl::NullRange, NULL, NULL);

		if (result != CL_SUCCESS) {
			std::cerr << "ERROR ENQUEUE KERNEL" << std::endl;
			throw result;
		}


		// leggo i risultati dell'operazione e li sposto in memoria host
		result = commandQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, vettoreC.size() * sizeof(float), vettoreC.data(), NULL);

		// Controllo che matrice finale sia matrice identità
		int ordine = sqrt(matriceA.size());
		int riga = 0; 
		for (int i = 0; i < vettoreC.size(); i++) {
			// Controllo che elemento su diagonale sia uguale ad 1
			if (i == (riga + riga*ordine)) {
				if((vettoreC[i] - 1) > 1e5){
					std::cout << "ERRORE, DIAGONALE DIVERSO DA 1" << std::endl;
					std::cout << vettoreC[i] << "!=" << 1 << std::endl;
					return;
				}
			}
			else {
				if(vettoreC[i] > 1e5){
					std::cout << "ERRORE, NON DIAGONALE DIVERSO DA 0" << std::endl;
					std::cout << vettoreC[i] << "!=" << 0 << std::endl;
					return;
				}
			}
			if (i != 0 && (i % ordine) == 0) {
				riga++;
			}
		}

		std::cout << "OK" << std::endl;

		// stampo risultato
		/*for (int i = 0; i < vettoreC.size(); i++) {
			if ( (i % ordine) == 0) {
				std::cout <<std::endl;
			}
			std::cout << vettoreC[i] << "\t\t";
		} */

		if (result != CL_SUCCESS) {
			std::cerr << "ERROR ENQUEUE READ BUFFER" << std::endl;
			throw result;
		}



	}
	catch (cl_int e) {

		std::cerr << "ERRORE: " << std::endl;
		std::cerr << e << std::endl;
	}






}