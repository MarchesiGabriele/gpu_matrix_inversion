#include <vector>
#include "headers.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <string>
#include "res_struct.h"
using namespace std;


int main(){
	#define N 3 
	#define REP 1 

	ofstream myFile;
	myFile.open("C:/TESI/TESI DOCUMENTAZIONE/results.txt");

	for (int k = 0; k < REP; k++) {
		if ((k%1 == 0 && k != 0) || N != 0) {
			//std::vector<double> matriceIniziale = std::vector<double>(k*k);
			//std::cout << "\n\nINDEX: " << k << std::endl;

			std::vector<double> matriceIniziale = std::vector<double>(N*N);
			std::cout << N << std::endl;


			Res matriceInversa;

			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale[i] = rand() % 10;
				//matriceIniziale[i] = (double)(1 / (double)(rand() % 1000 +1));
			}

			matriceIniziale = {4,8,8,2,4,5,5,1,7};

			if (k == 3) {
				for (int l = 0; l < matriceIniziale.size(); l++) {
					//matriceIniziale[i] = rand() % 10;
					if (l % k == 0 && l != 0)
						std::cout << std::endl;
					std::cout << matriceIniziale[l] << " ";
				}
			}


			matriceInversa = matrix_inversion(matriceIniziale, sqrt(matriceIniziale.size()));
			//matriceInversa = matrix_inversion_no_pivots(matriceIniziale, ordine);

			for (int i = 0; i < 10; i++) {
				myFile << matriceInversa.times[i] << " ";
			}

			double err = matrix_multiply(matriceInversa.inversa, matriceIniziale);

			if (err < 1e-10)
				myFile << "OK\n";
			else
				myFile << "ERRORE\n";

		}
	}

	myFile << "FINE";

	myFile.close();
}



