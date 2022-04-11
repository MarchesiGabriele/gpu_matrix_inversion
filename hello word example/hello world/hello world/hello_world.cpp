#include <CL/cl.h>
#include <CL/opencl.h>

int main() {
// //////////////////////////////////
/// SETUP
// Per prima cosa devo recuperare i dispositivi e le platform che sono a disposizione
	cl_device_id device = NULL;
	// Indico il tipo di dispositivo, il numero di dispositivi, il puntatore alla variabile per quel deispositivo
	clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device, NULL);

// Creo un context. Sia che comprenda più device sia che sia solo per un device. 
// Tutti i device devono far parte di un context
	// indico il numero di device nel context, il device
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

// Creo le command queues per inviare del lavoro ai dispositivi
	cl_command_queue queue = clCreateCommandQueue(context, device, (cl_command_queue_properties)0, NULL);

// //////////////////////////////////
/// COMPILATION
// Creo un programma
	// codice del kernel che verrà eseguito
	const char* source = {
		"kernel void calcSin(global float *data){\n"
		"	int id = get_global_id(0); \n"
		"	data[id] = sin(data[id]); \n"
		"}\n"
	};

// Compilo il programma 
// NB: in certi casi la compilazione è lazy, quindi viene solo eseuita quando un kernel è inviato in una command queue
	// indico il device, il numero di device nel context, ed un puntatore al source code del kernel
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, NULL);
	// preparo il programma per la compulazione (non è detto che avvenga in questo momento)
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);


// Creo i kernel usando il codice compilato, che poi saranno inviati alle command queues.
	// indico il programma creato ed il nome della funzione 
	cl_kernel kernel = clCreateKernel(program, "calcSin", NULL);
	

// //////////////////////////////////
/// CREATE MEMORY OBJECTS
	//Creo un oggetto di memoria (buffer o images)
	// Indico il context dei dispositivi che potranno accedere ai dati, indico le operazioni possibili sul dato, 
	// Indico le dimensioni dell'oggetto di memoria da allocare 
	size_t DATA_SIZE = 300;
	cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_SIZE, NULL, NULL);


// //////////////////////////////////
/// ENQUEUE WRITES 
// Copio i dati dalla cpu alla gpu
	const void* data;
	// vado a indicare la queue dove voglio mettere l'operazione di scrittura sul buffer specificato
	// data è un puntatore ai dati sulla cpu. I dati che voglio scrivere nel buffer
	clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, DATA_SIZE, data, 0, NULL, NULL);

// //////////////////////////////////
/// SET KERNEL ARGUMENTS
	// specifico il kernel a cui mi riferisco, l'index del parametro a cui mi riferisco, il valore del parametro che voglio passare
	// in questo caso prendo il buffer appena creato e lo passo come argomento al kernel
	clSetKernelArg(kernel, 1, 0, &buffer);

	
// //////////////////////////////////
/// ENQUEUE KERNEL FOR EXECUTION
// Metto il kernel in una command queue
 	size_t LENGTH;
	// imposto la dimensione globale per definire il numero di work items
	size_t global_dimensions[] = { LENGTH, 0, 0 };
	
	// metto in coda il kernel per essere poi eseguito
	// indico la coda dove lo metto, indico il kernel, indico la dimensione globale (1,2,3)
	// indico la dimensione locale (null in questo caso, lascio al dispositivo scelta automatica)
	// specifico le dimensioni globali, 
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_dimensions, NULL, 0, NULL, NULL);


// //////////////////////////////////
/// ENQUEUE READS
// Sposto i dati dalla gpu alla cpu
	// specifico queue e buffer, poi indico quanti dati voglio leggere, indico il puntatore del dato da cui partire a leggere
	clEnqueueReadBuffer(queue, buffer, CL_FALSE, 0, sizeof(cl_float) * LENGTH, data, 0, NULL, NULL);

// //////////////////////////////////
/// WAIT COMMANDS
// Aspetto che tutti i comandi vengano eseguiti 
// Poichè openCL è asincrono,devo aspettare che tutte le read, write e esecuzione dei kernel vengano completate
	clFinish(queue);

	





}