#include <vector>
#include "headers.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>


int main(){
	// Theoretical max is 16384
	#define N 2048 
	#define REP 1 
	#define WANTWRITEFILE false 

	// CREO MATRICE SU FILE TXT
	if (WANTWRITEFILE) {
		std::ofstream writeFile;
		writeFile.open("matrix.txt");

		std::vector<double> matriceIniziale = std::vector<double>(N*N);
		for (int i = 0; i < matriceIniziale.size(); i++) {
			matriceIniziale[i] = (double)(1 / (double)(rand() % 1000 +1));
			//matriceIniziale[i] = ((double)(rand() % 100));
		}

		for (int i = 0; i < matriceIniziale.size(); i++) {
			writeFile << matriceIniziale[i] << std::endl;
		}

		writeFile.close();
	}

	for (int k = 0; k < REP; k++) {
		std::ifstream readFile;
		readFile.open("matrix.txt");

		std::vector<double> matriceIniziale = std::vector<double>(N*N);
		std::vector<double> matriceInversa = std::vector<double>(N*N);


		// LEGGO MATRICE DA FILE TXT
		std::string line;
		int index = 0;
		while (std::getline (readFile, line)){
			matriceIniziale[index] = std::stod(line);
			index++;
		}
		readFile.close();
			

		//matriceIniziale = { 2,8,5,1,10,5,9,9,3 };
		//matriceIniziale = {2,8,5,1,10,5,9,9,3};
		//matriceIniziale = {1,1,1,1,1,1,1,1,1};
		//matriceIniziale = {1,0,0,0,0,0,0,0,1};


		int ordine = sqrt(matriceIniziale.size());

		// calcolo inversa
		matriceInversa = matrix_inversion(matriceIniziale, ordine);

		std::cout << std::setprecision(5) << "INVERSA: " << std::endl;

		std::cout << "\n\n PRIMI 50 VALORI MATRICE INVERSA" << std::endl;
		for (int i = 0; i < 50; i++) {
			std::cout << matriceInversa[i] <<  "  ";
		} 

		std::cout << "\n\n ULTIMI 50 VALORI MATRICE INVERSA" << std::endl;
		for (int i = matriceInversa.size()-50; i < matriceInversa.size(); i++) {
			std::cout << matriceInversa[i] <<  "  ";
		} 


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
