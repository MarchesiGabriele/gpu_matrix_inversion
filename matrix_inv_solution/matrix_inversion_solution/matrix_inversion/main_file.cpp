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
	#define N 4096 
	#define REP 500 
	#define PIVOTS true 
	#define RAND false 

	Res matriceInversa;
	
	if (RAND) {
		unsigned long j;
		srand((unsigned)time(NULL));
	}

	if (N == 0) {
		ofstream myFile;
		myFile.open("C:\\TESI\\TESI DOCUMENTAZIONE\\BENCHMARKS\\test.txt");
		//myFile.open("C:\\TESI\\TESI DOCUMENTAZIONE\\BENCHMARKS\\GJ_NOPIVOT_HOLLOW.txt");
	
		for (int k = 10; k < REP; k += 10) {
			std::vector<double> matriceIniziale = std::vector<double>(k * k);
			std::vector<double> matriceIniziale1 = std::vector<double>(k * k);
			std::vector<double> matriceInversa1 = std::vector<double>(k * k);
			std::cout << "\n\nINDEX: " << k << std::endl;

			// RIEMPIO MATRICE INIZIALE
		/*
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
			*/

			// MATRICE SPARSE
			for (int i = 0; i < matriceIniziale.size(); i++) {
				if ((int)(rand() % (matriceIniziale.size()/(k*k/5))) == 0)
					matriceIniziale[i] = (double)(rand() % 10);
				else	
					matriceIniziale[i] = 0;
			} 

/*
			for (int i = 0; i < matriceIniziale.size()-k; i++) {
				if (matriceIniziale[(int)(rand() % matriceIniziale.size())] != 0) {
					matriceIniziale[(int)(rand() % matriceIniziale.size())] = 0;
				}
				else {
					i--;
				}
			} 
*/
			int nonZeri = 0;
			for (int i = 0; i < matriceIniziale.size(); i++) {
				if (matriceIniziale[i] != 0)
					nonZeri++;
			} 

			std::cout << "Ordine: " << k << std::endl;
			std::cout << "Non Zeri: " << nonZeri << std::endl;



		
			// CALCOLO INVERSA
			//matriceInversa = FP32_bench(matriceIniziale, sqrt(matriceIniziale.size()));
			matriceInversa = FP64_bench(matriceIniziale, sqrt(matriceIniziale.size()));

			myFile << k << " ";
	/*	
			// CONVERTO DA FLOAT A DOUBLE
			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale1[i] = (double)(matriceIniziale[i]);
				matriceInversa1[i] = (double)(matriceInversa.inversa32[i]);
			}
	*/
		
			std::cout << matriceInversa.times.size() << std::endl;
			// SCRIVO SU FILE I TEMPI
			for (int i = 0; i <(PIVOTS ? 10 : 11); i++) {
				myFile << matriceInversa.times[i] << " ";
			}

			myFile << "\n";
			
			// ESEGUO CONTROLLO SOLO SE ORDINE MATRICE E' MINORE DI 8000
			if (k <= 8000) {
				matrix_multiply(matriceInversa.inversa64, matriceIniziale);

				if( k >= 2000)
					k += 990;
			}
			else {
				k += 990;
			}
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


