#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#define __CL_ENABLE_EXCEPTIONS


int main() {
	try {
		std::vector<cl::Platform> platform;
		std::vector<cl::Device> device;
		cl::Context context;
		cl::CommandQueue commandQueue;

		//std::vector<float> vettoreA = {1,2,3,4,5,6,7,8,9};
		std::vector<float> vettoreA(100000000, 11);
		int heightA = 10000;
		int widthA = 10000;

		//std::vector<float> vettoreB = {1,2,3,4,5,6,7,8,9};
		std::vector<float> vettoreB(100000000, 19);
		int heightB = 10000;
		int widthB = 10000;

		std::vector<float> vettoreC(heightA*widthB);
		cl_int result;

		/// SETUP
		// recupero piattaforma
		result =	cl::Platform::get(&platform);
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
		for (int i = 0; i < device.size(); i++) {
			std::cout << "DEVICE NAME: " << device[i].getInfo<CL_DEVICE_NAME>() << std::endl;
			std::cout << "MAX WORK GROUP DIMENSION: " << device[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
		}



		// il device al context	
		context = cl::Context(device[0]);

		// creo command queue
		commandQueue = cl::CommandQueue(context, device[0]);


		// CREAZIONE BUFFER E MOVIMENTO DATI IN MEMORIA
		// NB: i buffer che contengono delle matrici sono degli array di float!!
		cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vettoreA.size() * sizeof(float), vettoreA.data(), &result);
		cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vettoreB.size() * sizeof(float), vettoreB.data(), &result);
		cl::Buffer bufferC(context, CL_MEM_READ_ONLY, vettoreC.size() * sizeof(float), NULL, &result);
		if (result != CL_SUCCESS) {
			std::cerr << "ERROR CREATING BUFFERS" << std::endl;
			throw result;
		}

		
		// recupero kernel dal file esterno
		std::ifstream kernelFile("kernelFile.cl");
		std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

		// creo programma usando il kernel
		cl::Program program(context, cl::Program::Sources(1, std::make_pair(src.c_str(), src.length() + 1)), &result);

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
		result = kernelMultiply.setArg(1, widthA);
		result = kernelMultiply.setArg(2, heightA);
		result = kernelMultiply.setArg(3, widthB);
		result = kernelMultiply.setArg(4, heightB);
		result = kernelMultiply.setArg(5, bufferA);
		result = kernelMultiply.setArg(6, bufferB);

		if (result != CL_SUCCESS) {
			std::cerr << "ERROR SETTING ARGUMENTS" << std::endl;
			throw result;
		}


		// TODO: creo eventuali workgroups e utilizzo se serve la memoria locale per ottimizzare esecuzione

			
		// eseguo il kernel
		//NB: NDrange sono le dimensioni globali e locali
		result = commandQueue.enqueueNDRangeKernel(kernelMultiply, cl::NullRange, cl::NDRange(widthB, heightA), cl::NullRange, NULL, NULL);

		if (result != CL_SUCCESS) {
			std::cerr << "ERROR ENQUEUE KERNEL" << std::endl;
			throw result;
		}


		// leggo i risultati dell'operazione e li sposto in memoria host
		result = commandQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, vettoreC.size() * sizeof(float), vettoreC.data(), NULL);

		if (result != CL_SUCCESS) {
			std::cerr << "ERROR ENQUEUE READ BUFFER" << std::endl;
			throw result;
		}



	}
	catch (cl_int e){
		
		std::cerr << "ERRORE: " << std::endl;
		std::cerr << e << std::endl;
	}






}