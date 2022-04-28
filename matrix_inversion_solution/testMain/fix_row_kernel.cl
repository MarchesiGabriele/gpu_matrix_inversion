/*
Eseguo questo kernel per ciascuna riga della matrice augmentata 
l iterazione della matrice viene eseguita dal progetto c++ 
il problema è quindi in una sola dimensione

Prendo ciascun elemento della riga fino ad arrivare a metà della matrice partendo da sx 
Copio la riga intera della matrice e la metto dentro un vettore riga
Assegno ad ogni thread un elemento della riga
Ogni elemento viene diviso per Aii ovvero lelemento sulla diagonale 
*/

__kernel void fixRowKernel(__global float *matrix, int size, int rowId){

	__local float row[100];

	__local float Aii;

	int colId = get_global_id(0);

	row[colId] = matrix[size*rowId + colId];
	Aii = matrix[size*rowId + rowId];

	row[colId] = row[colId]/Aii;
	matrix[size*rowId + colId] = row[colId];
}






