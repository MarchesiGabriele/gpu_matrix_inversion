#include <vector>
#include "headers.h"
#include <iostream>



int main(){
	// Theoretical max is 16384
	#define N 3 
	#define REP 1

	for (int k = 0; k < REP; k++) {
		std::vector<float> matriceIniziale = std::vector<float>(N*N);
		// riempio matrice iniziale
		for (int i = 0; i < matriceIniziale.size(); i++) {
			matriceIniziale[i] = rand() % 10 + 1;
		}

		int ordine = sqrt(matriceIniziale.size());

		// calcolo inversa
		std::vector<float> matriceInversa =  matrix_inversion(matriceIniziale, ordine);

		// controllo che inversa sia corretta 
		matrix_multiply(matriceIniziale, matriceInversa);
	}
}



/*
	// Theoretical max is 16384
	#define N 3 
	#define REP 1

	for (int k = 0; k < REP; k++) {
		std::vector<float> matriceIniziale = std::vector<float>(N*N);
		for (int i = 0; i < matriceIniziale.size(); i++) {
			matriceIniziale[i] = rand() % 10 + 1;
			std::cout << matriceIniziale[i] << " ";
		}

		int ordine = sqrt(matriceIniziale.size());

		std::cout << std::endl;
		std::cout << std::endl;

		// calcolo inversa
		std::vector<float> matriceInversa =  matrix_inversion(matriceIniziale, ordine);

		for (int i = 0; i <matriceInversa.size(); i++) {
			std::cout << matriceInversa[i] << " ";
		}
		
		std::cout << std::endl;
		std::cout << std::endl;

		// controllo che inversa sia corretta 
		matrix_multiply(matriceIniziale, matriceInversa);
	}
*/
