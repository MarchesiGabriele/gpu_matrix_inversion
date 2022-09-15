#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#define __CL_ENABLE_EXCEPTIONS
using namespace std::chrono;



// NB: posso moltiplicare solamente matrici quadrate della stessa dimensione
void matrix_multiply(std::vector<float> matriceB, std::vector<float> matriceA) {
	try {
		const std::string kernel = R"(
			#pragma OPENCL EXTENSION cl_khr_fp64 : enable
			__kernel void simpleMultiply(
				__global float* outputC,
				int ordine,
				__global float* inputA,
				__global float* inputB) {
			int row = get_global_id(1);
			int col = get_global_id(0);

			float sum = 0.0;

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

		std::vector<cl_float> vettoreC((cl_float)(sqrt(matriceA.size()) * sqrt(matriceB.size())));
		cl_int result;

		/// SETUP
		// recupero piattaforma
		result = cl::Platform::get(&platform);
		if (result != CL_SUCCESS) {
			std::cerr << "ERROR GETTING PLATFORM" << std::endl;
			throw result;
		}

		// recupero device 
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
		std::vector<cl::Buffer> buffers;
		buffers.push_back(cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matriceA.size() * sizeof(cl_float), matriceA.data(), &result));
		buffers.push_back(cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matriceB.size() * sizeof(cl_float), matriceB.data(), &result));
		buffers.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, vettoreC.size() * sizeof(cl_float), NULL, &result));
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
		result = kernelMultiply.setArg(0, buffers[2]);
		result = kernelMultiply.setArg(1, (cl_int)sqrt(matriceA.size()));
		result = kernelMultiply.setArg(2, buffers[0]);
		result = kernelMultiply.setArg(3, buffers[1]);

		if (result != CL_SUCCESS) {
			std::cerr << "ERROR SETTING ARGUMENTS" << std::endl;
			throw result;
		}



		// eseguo il kernel
		//NB: NDrange sono le dimensioni globali e locali
		steady_clock::time_point inizio = steady_clock::now();
		result = commandQueue.enqueueNDRangeKernel(kernelMultiply, cl::NullRange, cl::NDRange((cl_int)sqrt(matriceA.size()), (cl_int)sqrt(matriceA.size())), cl::NullRange, NULL, NULL);

		if (result != CL_SUCCESS) {
			std::cerr << "ERROR ENQUEUE KERNEL" << std::endl;
			throw result;
		}


		result = commandQueue.finish();
		steady_clock::time_point fine = steady_clock::now();

		duration<float> time = duration_cast<duration<float>> (fine - inizio);




		result = commandQueue.enqueueNDRangeKernel(kernelMultiply, cl::NullRange, cl::NDRange((cl_int)sqrt(matriceA.size()), (cl_int)sqrt(matriceA.size())), cl::NullRange, NULL, NULL);
		if (result != CL_SUCCESS) {
			std::cerr << "ERROR GETTING DEVICES" << std::endl;
			throw result;
		}

		commandQueue.finish();

		// leggo i risultati dell'operazione e li sposto in memoria host
		result = commandQueue.enqueueReadBuffer(buffers[2], CL_TRUE, 0, vettoreC.size() * sizeof(cl_float), vettoreC.data(), NULL);


		if (result != CL_SUCCESS) {
			std::cerr << "ERROR GETTING DEVICES" << std::endl;
			throw result;
		}


		if (result != CL_SUCCESS) {
			std::cerr << "ERROR SETTING ARGUMENTS" << std::endl;
			throw result;
		}



		int tot = 0;
		for (int i = 0; i < vettoreC.size(); i++) {
			if ((int)vettoreC[i] > 0) {
				tot++;
			}
		}


		/*for (int i = 0; i < vettoreC.size(); i++) {
			if ((int)vettoreC[i] >0) {
				std::cout << "POSIZIONE: " << i << " , VALORE: " << vettoreC[i] << std::endl;
			}
		} */

		int ordine = sqrt(vettoreC.size());
		/*	int r = 0;
			for (int i = 0; i < vettoreC.size(); i++) {
				if (i != 0 && (i % ordine) == 0) {
					r++;
				}
				if (i == (r * ordine + r)) {
					vettoreC[i] = 1 - vettoreC[i];
				}
			}
		*/



		// NORMA DI FROBENIUS
		float somma = 0.0;
		for (int i = 0; i < vettoreC.size(); i++) {
			somma += vettoreC[i] * vettoreC[i];
			//std::cout << "\n\nSOMMA ATTUALE: " << somma << std::endl;
		}
		/*
				if (somma > 1) {
					// STAMPO PORZIONE DEL VETTORE C
					for (int i = 0; i < vettoreC.size()/100; i++) {
						std::cout << vettoreC[i] <<  "  ";
					}

				}
		*/


		auto errore = sqrt(ordine) - sqrt(somma);

		std::cout << std::setprecision(60) << "\nERRORE: " << errore << std::endl;
		std::cout << std::setprecision(60) << "\nORDINE: " << sqrt(ordine) << std::endl;
		std::cout << std::setprecision(60) << "\nSOMMA: " << sqrt(somma) << std::endl;

		buffers.clear();
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