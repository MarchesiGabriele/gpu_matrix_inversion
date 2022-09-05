#include <vector>
#include "headers.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
using namespace std;


int main(){
	// Theoretical max is 16384
	#define N 130 
	#define REP 100

	ofstream myFile;

	myFile.open("C:\\TESI\\TESI DOCUMENTAZIONE\\results.txt");

	for (int k = 0; k < REP; k++) {
		std::vector<double> matriceIniziale = std::vector<double>(N*N);
		std::vector<double> matriceInversa = std::vector<double>(N*N);

		for (int i = 0; i < matriceIniziale.size(); i++) {
			matriceIniziale[i] = rand() % 10;
			//matriceIniziale[i] = (double)(1 / (double)(rand() % 1000 +1));
		}

		//matriceIniziale = {2,8,5,1,10,5,9,9,3 };
		//matriceIniziale = {0.01232,0.0012,0.12,0.998,0.007,0.00542,0.01,0.00433, 0.9};
		//matriceIniziale = {9,8,5,1,10,5,9,9,3};
		//matriceIniziale = {4,8,8,2,4,5,5,1,7};
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
		

		// calcolo inversa
		matriceInversa = matrix_inversion(matriceIniziale, ordine);
		//matriceInversa = matrix_inversion_no_pivots(matriceIniziale, ordine);

		// controllo che inversa sia corretta 
		
		double err = matrix_multiply(matriceInversa, matriceIniziale);
		myFile << err;

		if (err < 1e-10)
			myFile << "\t\tOK\n";
		else
			myFile << "\t\tERRORE\n";
	}
	myFile.close();
}



