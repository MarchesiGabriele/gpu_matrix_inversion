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
		// 10 righe 5 colonne
		std::vector<float> vettoreA(50, 3);
		// 5 righe 10 colonne 
		std::vector<float> vettoreB(50, 3);
		// 10 x 10 
		std::vector<float> vettoreC(100);


		/// SETUP
		// recupero piattaforma
		cl::Platform::get(&platform);

		// recupero device 
		// TODO: CAPIRE AUTOMATICAMENTE COME PRENDERE LA GPU FISICA E NON QUELLA INTEGRATA DEL PROCESSORE
		platform[0].getDevices(CL_DEVICE_TYPE_ALL, &device);
		for (int i = 0; i < device.size(); i++) {
			std::cout << device[i].getInfo<CL_DEVICE_NAME>() << std::endl;
		}

		// il device al context	
		context = cl::Context(device[0]);

		// creo command queue
		commandQueue = cl::CommandQueue(context, device[0]);


		// CREAZIONE BUFFER E MOVIMENTO DATI IN MEMORIA
		// NB: i buffer che contengono delle matrici sono degli array di float!!
		cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vettoreA.size() * sizeof(float), vettoreA.data());
		cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vettoreB.size() * sizeof(float), vettoreB.data());
		cl::Buffer bufferC(context, CL_MEM_READ_ONLY, vettoreC.size() * sizeof(float), vettoreC.data());

		// scrivo dati nella memoria nei buffer appena creati
		cl::enqueueWriteBuffer(bufferA, CL_TRUE, 0, vettoreA.size() * sizeof(float), vettoreA.data(), 0, NULL);
		cl::enqueueWriteBuffer(bufferB, CL_TRUE, 0, vettoreB.size() * sizeof(float), vettoreB.data(), 0, NULL);

		// recupero kernel dal file esterno
		std::ifstream kernelFile("kernelFile.cl");
		std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

		// creo programma usando il kernel
		cl::Program program(context, cl::Program::Sources(1, std::make_pair(src.c_str(), src.length() + 1)));

		// compilo il programma
		try {
			program.build(device);
		}
		catch (...) {
			std::cerr << "ERRORE " << std::endl;
			return 1;
		}

		// creo kernel
		cl::Kernel kernelMultiply(program, "simpleMultiply");

		// imposto gli argomenti del kernel	
		kernelMultiply.setArg(0, bufferC);
		kernelMultiply.setArg(1, 5);
		kernelMultiply.setArg(2, 10);
		kernelMultiply.setArg(3, 10);
		kernelMultiply.setArg(4, 5);
		kernelMultiply.setArg(5, bufferA);
		kernelMultiply.setArg(6, bufferB);

		// creo eventuali workgroups e utilizzo se serve la memoria locale per ottimizzare esecuzione

		// eseguo il kernel
		//NB: NDrange sono le dimensioni globali e locali
		commandQueue.enqueueNDRangeKernel(kernelMultiply, 0, 2, NULL, NULL);


		// leggo i risultati dell'operazione e li sposto in memoria host
		commandQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, vettoreC.size() * sizeof(float), vettoreC.data(), NULL);

		for (int i = 0; i < vettoreC.size(); i++) {
			std::cout << vettoreC[i] << std::endl;
		}
	}
	catch (...) {
		std::cerr << "ERROR BIG JUICE" << std::endl;
	}






}