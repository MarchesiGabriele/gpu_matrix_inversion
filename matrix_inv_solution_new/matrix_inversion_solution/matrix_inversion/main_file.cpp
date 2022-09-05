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
#define N 0 
#define REP 2048 
#define STEP 10

	ofstream myFile;
	myFile.open("C:\\TESI\\TESI DOCUMENTAZIONE\\BENCHMARKS\\GJ_PIVOT_KERNEL_TIMES.txt");
	//myFile.open("C:\\TESI\\TESI DOCUMENTAZIONE\\BENCHMARKS\\GJ_PIVOT_NO_KERNEL_TIMES.txt");
	Res matriceInversa;

	if (N == 0) {
		for (int k = STEP; k < REP; k += STEP) {
			std::vector<double> matriceIniziale = std::vector<double>(k * k);
			std::cout << "\n\nINDEX: " << k << std::endl;

			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale[i] = rand() % 10;
				//matriceIniziale[i] = (double)(1 / (double)(rand() % 1000 +1));
			}

			matriceInversa = matrix_inversion_bench(matriceIniziale, sqrt(matriceIniziale.size()));
			//matriceInversa = matrix_inversion_no_pivots_bench(matriceIniziale, ordine);

			myFile << k << " ";
			for (int i = 0; i < 10; i++) {
				myFile << matriceInversa.times[i] << " ";
			}

			double err = matrix_multiply(matriceInversa.inversa, matriceIniziale);

			if (abs(err) < 1e-10)
				myFile << "OK\n";
			else {
				myFile << "ERRORE\n";
			}
		}
	}
	else {
		std::vector<double> matriceIniziale = std::vector<double>(N*N);
		std::cout << N << std::endl;

		for (int i = 0; i < matriceIniziale.size(); i++) {
			matriceIniziale[i] = rand() % 10;
			//matriceIniziale[i] = (double)(1 / (double)(rand() % 1000 +1));
		}

		matriceInversa = matrix_inversion_bench(matriceIniziale, sqrt(matriceIniziale.size()));
		//matriceInversa = matrix_inversion_no_pivots_bench(matriceIniziale, ordine);

		double err = matrix_multiply(matriceInversa.inversa, matriceIniziale);
	}

	myFile << "FINE";
	myFile.close();
}


