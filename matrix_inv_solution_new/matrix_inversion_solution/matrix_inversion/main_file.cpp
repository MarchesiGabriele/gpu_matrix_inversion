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
	#define N 0 
	#define REP 2048 

	ofstream myFile;
	myFile.open("C:\\TESI\\TESI DOCUMENTAZIONE\\results.txt");
	Res matriceInversa;

	for (int k = 0; k < REP; k++) {
		if ((k%10 == 0 && k != 0) || N != 0) {
			std::vector<double> matriceIniziale = std::vector<double>(k*k);
			std::cout << "\n\nINDEX: " << k << std::endl;

			//std::vector<double> matriceIniziale = std::vector<double>(N*N);
			//std::cout << N << std::endl;

			for (int i = 0; i < matriceIniziale.size(); i++) {
				matriceIniziale[i] = rand() % 10;
				//matriceIniziale[i] = (double)(1 / (double)(rand() % 1000 +1));
			}

			matriceInversa = matrix_inversion(matriceIniziale, sqrt(matriceIniziale.size()));
			//matriceInversa = matrix_inversion_no_pivots(matriceIniziale, ordine);
		
			myFile << k << " ";
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


