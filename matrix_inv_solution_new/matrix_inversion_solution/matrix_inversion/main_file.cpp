#include <vector>
#include "headers.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <string>
#include "res_struct.h"
using namespace std;


int main() {
#define FP32 true 
#define N 4096 
#define REP 16000 
#define PIVOTS false 

Res matriceInversa;

	if (N == 0) {
		/*
		ofstream myFile;
		//myFile.open("C:\\TESI\\TESI DOCUMENTAZIONE\\BENCHMARKS\\GJ_PIVOT_KERNEL_TIMES.txt");
		//myFile.open("C:\\TESI\\TESI DOCUMENTAZIONE\\BENCHMARKS\\GJ_PIVOT_NO_KERNEL_TIMES.txt");
		myFile.open("C:\\TESI\\TESI DOCUMENTAZIONE\\BENCHMARKS\\GJ_NO_PIVOT_KERNEL_TIMES.txt");
		//myFile.open("C:\\TESI\\TESI DOCUMENTAZIONE\\BENCHMARKS\\GJ_NO_PIVOT_NO_KERNEL_TIMES.txt");
	
		for (int k = 10; k < REP; k += 10) {
			std::vector<double> matriceIniziale = std::vector<double>(k * k);
			std::cout << "\n\nINDEX: " << k << std::endl;

			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale[i] = rand() % 10;
			}

			//matriceInversa = matrix_inversion_bench(matriceIniziale, sqrt(matriceIniziale.size()));
			matriceInversa = matrix_inversion_no_pivots_bench(matriceIniziale,sqrt(matriceIniziale.size()));

			myFile << k << " ";

			if (PIVOTS) {
				for (int i = 0; i < 10; i++) {
					myFile << matriceInversa.times[i] << " ";
				}

				if (k <= 8000) {
					double err = matrix_multiply(matriceInversa.inversa, matriceIniziale);
					if (abs(err) < 1e-10)
						myFile << "OK\n";
					else 
						myFile << "ERRORE\n";

					if( k >= 2000)
						k += 990;
				}
				else {
					k += 990;
					myFile << "OKK\n";
				}
			}
			else {
				for (int i = 0; i < 11; i++) {
					myFile << matriceInversa.times[i] << " ";
				}

				if (k <= 8000) {
					double err = matrix_multiply(matriceInversa.inversa, matriceIniziale);
					if (abs(err) < 1e-10)
						myFile << "OK\n";
					else
						myFile << "ERRORE\n";
				
					if (k >= 2000)
						k += 990;
				}
				else {
					k += 990;
					myFile << "OKK\n";
				} 
			}
		}
		myFile << "FINE";
		myFile.close();
		*/
	}
	else {
		if(FP32){
			std::vector<float> matriceIniziale = std::vector<float>(N*N);
			std::vector<float> matriceInversa = std::vector<float>(N*N);
			std::cout << N << std::endl;

			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale[i] = (float)(rand() % 10);
				//matriceIniziale[i] = (double)(1 / (double)(rand() % 1000 +1));
			}

			matriceInversa = matrix_inversion_FP32(matriceIniziale, sqrt(matriceIniziale.size()));
			//matriceInversa = matrix_inversion_no_pivots_bench(matriceIniziale, sqrt(matriceIniziale.size()));
			/*
			for (int i = 0; i < 10; i++) {
				std::cout << matriceInversa.times[i] << " ";
			}
			*/

			matrix_multiply(matriceInversa, matriceIniziale);
		}
		else {
			std::vector<double> matriceIniziale = std::vector<double>(N*N);
			std::vector<double> matriceInversa = std::vector<double>(N*N);
			std::vector<float> matriceIniziale1 = std::vector<float>(N*N);
			std::vector<float> matriceInversa1 = std::vector<float>(N*N);

			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale[i] = (double)(rand() % 10);
				//matriceIniziale[i] = (double)(1 / (double)(rand() % 1000 +1));
			}

			matriceInversa = matrix_inversion_FP64(matriceIniziale, sqrt(matriceIniziale.size()));

			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale1[i] = (float)(matriceIniziale[i]);
				matriceInversa1[i] = (float)(matriceInversa[i]);
			}


			matrix_multiply(matriceInversa1, matriceIniziale1);
		}
	
	}

}


