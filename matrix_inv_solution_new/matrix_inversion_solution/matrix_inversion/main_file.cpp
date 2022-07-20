#include <vector>
#include "headers.h"
#include <iostream>
#include <iomanip>



int main(){
	// Theoretical max is 16384
	#define N 3 
	#define REP 1

	for (int k = 0; k < REP; k++) {
		std::vector<double> matriceIniziale = std::vector<double>(N*N);
		std::vector<double> matriceInversa = std::vector<double>(N*N);
		std::cout << "INIZIALE: " << std::endl;
		for (int i = 0; i < matriceIniziale.size(); i++) {
			matriceIniziale[i] = (double)(rand() % 1000) + 1;
			//std::cout << matriceIniziale[i] << " ";
		}
		//matriceIniziale = {2,1,5,4,2,7,8,9,3};
		matriceIniziale = {2,8,5,1,10,5,9,9,3};

		int ordine = sqrt(matriceIniziale.size());

		// calcolo inversa
		matriceInversa = matrix_inversion(matriceIniziale, ordine);

		std::cout << std::setprecision(60) << "INVERSA: " << std::endl;

		for (int i = 0; i <matriceInversa.size(); i++) {
			if (i != 0 && (i % N) == 0) {
				std::cout << std::endl;
			}
			std::cout << matriceInversa[i] << "\t\t";
		}

		std::cout  << std::endl;
		std::cout << std::endl;

		// controllo che inversa sia corretta 
		matrix_multiply(matriceInversa, matriceIniziale);
	}
}



/*
	// Theoretical max is 16384
	#define N 10 
	#define REP 1

	for (int k = 0; k < REP; k++) {
		std::vector<float> matriceIniziale = std::vector<float>(N*N);
		std::cout << "INIZIALE: " << std::endl;
		for (int i = 0; i < matriceIniziale.size(); i++) {
			matriceIniziale[i] = rand() % 10 + 1;
			std::cout << matriceIniziale[i] << " ";
		}

		int ordine = sqrt(matriceIniziale.size());

		std::cout << std::endl;
		std::cout << std::endl;

		// calcolo inversa
		std::vector<float> matriceInversa =  matrix_inversion(matriceIniziale, ordine);

		std::cout << "INVERSA: " << std::endl;
		for (int i = 0; i <matriceInversa.size(); i++) {
			std::cout << matriceInversa[i] << " ";
		}
		
		std::cout << std::endl;
		std::cout << std::endl;

		// controllo che inversa sia corretta 
		matrix_multiply(matriceIniziale, matriceInversa);
	}
*/
