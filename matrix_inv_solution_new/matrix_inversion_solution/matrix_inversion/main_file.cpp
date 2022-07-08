#include <vector>
#include "headers.h"
#include <iostream>

int main(){
		std::vector<float> matriceIniziale = {1,3,5,2,4,3,2,3,4};
		int ordine = sqrt(matriceIniziale.size());
		// calcolo inversa
		std::vector<float> matriceInversa =  matrix_inversion(matriceIniziale, ordine);

		// controllo che inversa sia corretta 
		matrix_multiply(matriceIniziale, matriceInversa);
	
		// stampo matrice inversa
		for (int i = 0; i < matriceInversa.size(); i++) {
			if ( i % ordine == 0) {
				std::cout <<std::endl;
			}
			std::cout << matriceInversa[i] << "\t\t";
		}
	}
