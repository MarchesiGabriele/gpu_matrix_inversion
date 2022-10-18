#include <vector>
#include "headers.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <time.h>
#include <string>
#include "res_struct.h"
using namespace std;


int main() {
	#define FP32 true 
	#define N 0 
	#define REP 16000
	#define PIVOTS false 
	#define RAND false 

	Res matriceInversa;
	
	if (RAND) {
		unsigned long j;
		srand((unsigned)time(NULL));
	}

	if (N == 0) {
		ofstream myFile;
		myFile.open("C:\\Users\\march\\Desktop\\TESI DOCUMENTAZIONE\\BENCHMARKS\\errore_64.txt");
	
		for (int k = 10; k < REP; k += 10) {
			//std::vector<float> matriceIniziale = std::vector<float>(k * k);
			std::vector<double> matriceIniziale = std::vector<double>(k * k);
			std::vector<double> matriceIniziale1 = std::vector<double>(k * k);
			std::vector<double> matriceInversa1 = std::vector<double>(k * k);
			std::cout << "\n\nINDEX: " << k << std::endl;


			myFile << k << " ";
			// RIEMPIO MATRICE INIZIALE
			int riga = 0;
			for (int i = 0; i < matriceIniziale.size(); i++) {
				if (i == (k * riga)) {
					riga++;
				}
				if (i == (k * (riga-1)+(riga-1))) {
					matriceIniziale[i] = 0;
				}
				else {
					matriceIniziale[i] = rand() % 10;
				}
			}

			// CALCOLO INVERSA
			//matriceInversa = FP32_bench(matriceIniziale, sqrt(matriceIniziale.size()));
			matriceInversa = FP64_bench(matriceIniziale, sqrt(matriceIniziale.size()));

			// CONVERTO DA FLOAT A DOUBLE
			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale1[i] = (double)(matriceIniziale[i]);
				matriceInversa1[i] = (double)(matriceInversa.inversa64[i]);
			}
		
			std::cout << matriceInversa.times.size() << std::endl;

			// SCRIVO SU FILE I TEMPI
			/*for (int i = 0; i <(PIVOTS ? 10 : 11); i++) {
				myFile << matriceInversa.times[i] << " ";
			}
			*/
			
			// ESEGUO CONTROLLO SOLO SE ORDINE MATRICE E' MINORE DI 8000
			if (k <= 8000) {
				if( k >= 2000) k += 990;
			}
			else {
				k += 990;
			}

			auto errore = matrix_multiply(matriceInversa1, matriceIniziale1);
			myFile << errore << std::endl;
		}
		myFile.close();
	}
	else {
		if(FP32){
			std::vector<float> matriceIniziale = std::vector<float>(N*N);
			std::vector<double> matriceIniziale1 = std::vector<double>(N*N);
			std::vector<double> matriceInversa1 = std::vector<double>(N*N);
			std::vector<float> matriceInversa = std::vector<float>(N*N);
			//Res matriceInversa;

			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale[i] = (float)(rand() % 10);
			}

			//matriceInversa = FP32_bench(matriceIniziale, sqrt(matriceIniziale.size()));
			matriceInversa = matrix_inversion_FP32(matriceIniziale, sqrt(matriceIniziale.size()));
/*
			for (int i = 0; i < 10; i++) {
				std::cout << matriceInversa.times[i] << " ";
			}
*/
			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale1[i] = (double)(matriceIniziale[i]);
				matriceInversa1[i] = (double)(matriceInversa[i]);
			}
			matrix_multiply(matriceInversa1, matriceIniziale1);
		}
		else {
			std::vector<double> matriceIniziale = std::vector<double>(N*N);
			std::vector<double> matriceInversa = std::vector<double>(N*N);

			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale[i] = (double)(rand() % 10);
				//matriceIniziale[i] = (double)(1 / (double)(rand() % 1000 +1));
			}

			//matriceInversa = matrix_inversion_FP64(matriceIniziale, sqrt(matriceIniziale.size()));
			matriceInversa = matrix_inversion_no_pivots(matriceIniziale, sqrt(matriceIniziale.size()));

		
			matrix_multiply(matriceInversa, matriceIniziale);
		}
	
	}

}


