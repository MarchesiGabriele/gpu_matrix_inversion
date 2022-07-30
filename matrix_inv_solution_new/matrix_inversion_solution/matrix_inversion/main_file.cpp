#include <vector>
#include "headers.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>


int main(){
	// Theoretical max is 16384
	#define N 4096 
	#define REP 1 
	#define WANTWRITEFILE false 

	// CREO MATRICE SU FILE TXT
	if (WANTWRITEFILE) {
		std::ofstream writeFile;
		writeFile.open("matrix.txt");

		std::vector<double> matriceIniziale = std::vector<double>(N*N);
		for (int i = 0; i < matriceIniziale.size(); i++) {
			matriceIniziale[i] = (double)(1 / (double)(rand() % 10 +1));
			//matriceIniziale[i] = ((double)(rand() % 100));
		}

		for (int i = 0; i < matriceIniziale.size(); i++) {
			writeFile << matriceIniziale[i] << std::endl;
		}

		writeFile.close();
	}

	for (int k = 0; k < REP; k++) {
		std::vector<double> matriceIniziale = std::vector<double>(N*N);
		std::vector<double> matriceInversa = std::vector<double>(N*N);

/*
		// LEGGO MATRICE DA FILE TXT
		std::ifstream readFile;
		readFile.open("matrix.txt");
		std::string line;
		int index = 0;
		while (std::getline (readFile, line)){
			matriceIniziale[index] = std::stod(line);
			index++;
		}
		readFile.close();
*/
		for (int i = 0; i < matriceIniziale.size(); i++) {
			//matriceIniziale[i] = rand() % 10;
			matriceIniziale[i] = (double)(1 / (double)(rand() % 1000 +1));
		}

	

		//matriceIniziale = {2,8,5,1,10,5,9,9,3 };
		//matriceIniziale = {0.01232,0.0012,0.12,0.998,0.007,0.00542,0.01,0.00433, 0.9};
		//matriceIniziale = {9,8,5,1,10,5,9,9,3};
		//matriceIniziale = {1,1,1,1,1,1,1,1,1};
		//matriceIniziale = {1,0,0,0,0,0,0,0,1};
/*
		// ILL CONDITIONED MATRIX
		matriceIniziale.clear();
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				matriceIniziale.push_back((double)1/(i+j+1));
			}
		}
*/
/*
		for (int i = 0; i < N*N; i++) {
			std::cout << matriceIniziale[i] << " ";
		}
*/

		int ordine = sqrt(matriceIniziale.size());
/*
		std::vector<double> matrice_augmentata = {};
		int riga = 0;
		for (int i = 0; i < (matriceIniziale.size() * 2); i++) {
			if (i == (riga * ordine* 2)) {
				riga++;
			}

			if (i < (ordine * riga + ordine*(riga-1))) {
				matrice_augmentata.push_back(matriceIniziale[i - ((riga-1) * ordine)]);
			}
			else {
				if (i == (riga * ordine + (riga-1)*ordine + riga-1)){
					matrice_augmentata.push_back(1);
				}else{
					matrice_augmentata.push_back(0);
				}
			}
		}
*/
	/*
		std::cout << "\nINIZIALE: " << std::endl;
		for (int i = 0; i < 256; i++) {
			if (i != 0 && (i % (ordine * 2)) == 0) {
				std::cout << std::endl;	
			}
		
			std::cout << matrice_augmentata[i*512] << " ";
		} pivot_max_test(matrice_augmentata, ordine);
*/





		

		// calcolo inversa
		matriceInversa = matrix_inversion(matriceIniziale, ordine);
/*
		std::cout << std::setprecision(5) << "INVERSA: " << std::endl;

		std::cout << "\n\n PRIMI 50 VALORI MATRICE INVERSA" << std::endl;
		for (int i = 0; i < 50; i++) {
			std::cout << matriceInversa[i] <<  "  ";
		} 

		std::cout << "\n\n ULTIMI 50 VALORI MATRICE INVERSA" << std::endl;
		for (int i = matriceInversa.size()-50; i < matriceInversa.size(); i++) {
			std::cout << matriceInversa[i] <<  "  ";
		} 
*/

/*
		for (int i = 0; i <matriceInversa.size(); i++) {
			if (i != 0 && (i % N) == 0) {
				std::cout << std::endl;
			}
			std::cout << matriceInversa[i] << "\t\t";
		}
		std::cout  << std::endl;
		std::cout << std::endl;
*/

		// controllo che inversa sia corretta 
		matrix_multiply(matriceInversa, matriceIniziale);
	}
}



