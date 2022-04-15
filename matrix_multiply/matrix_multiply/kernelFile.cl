/*
NB: outputC, inputA, inputB sono dei buffer presenti in memoria globale i cui elementi sono tutti float!! 

Parametro1: puntatore alla matrice output. __global indica che si trova nella
memoria globale. 

Parametro6 e Parametro7: Puntatori alle matrici di input
*/

__kernel void simpleMultiply(
	__global float *outputC,
	int widthA,
	int heightA,
	int widthB,
	int heightB,
	__global float *inputA,
	__global float *inputB){

	
	/* recupero index di riga e colonna del singolo work item  */
	/* serve per sapere se sono sulla riga/colonna 0 oppure 1  */
	int row = get_global_id(1);
	int col = get_global_id(0);

	float sum = 0.0f;
	
	/* NB: inputA e inputB sono matrici! ma in memoria corrispondono a dei
vettori. Ogni work item si prende una linea ed una colonna */		

	for(int i = 0; i<widthA; i++){
		sum += inputA[row*widthA +i] * inputB[i*widthB + col];
	} 

	outputC[row*widthB +col] = sum; 

}










