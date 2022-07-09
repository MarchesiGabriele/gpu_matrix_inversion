#include <vector>
#include "headers.h"
#include <iostream>



int main(){
	#define N 1000 
	//std::vector<float> matriceIniziale = {1,3,5,2,4,3,2,3,4};
	
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
