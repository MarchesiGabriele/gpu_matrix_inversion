__kernel void simpleMultiply(
	__global float *outputC,
	int widthA,
	int heightA,
	int widthB,
	int heightB,
	__global float *inputA,
	__global float *inputB){

	int row = get_global_id(1);
	int col = get_global_id(0);

	outputC[row*widthB +col] = 6;
}
