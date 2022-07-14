#include <vector>
#include "headers.h"
#include <iostream>



int main(){
	// Theoretical max is 16384
	#define N 4096 
	#define REP 1

	for (int i = 0; i < REP; i++) {
		std::vector<float> matriceIniziale = std::vector<float>(N*N);
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
