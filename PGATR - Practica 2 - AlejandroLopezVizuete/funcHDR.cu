#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

//Kernel para Exclusive Scan
__global__
void scan(unsigned int * d_histograma, int tam) {
	//Memoria compartida
	__shared__ int tempArray[1024 * 2];
	// Seleccionamos el thread correspondiente
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int threadId = threadIdx.x;
	//Creamos todas las variables necesarias
	int offset = 1, temp;
	int ai = threadId;
	int bi = threadId + tam / 2;
	int i;
	//Guardamos en la memoria compartida los datos del histograma
	tempArray[ai] = d_histograma[id];
	tempArray[bi] = d_histograma[id + tam / 2];

	//PARTE 1 REDUCCION
	//Recorremos los datos del histograma para conseguir las sumas parciales
	for (i = tam >> 1; i > 0; i >>= 1) {
		__syncthreads();
		if (threadId < i)
		{
			ai = offset*(2 * threadId + 1) - 1;
			bi = offset*(2 * threadId + 2) - 1;
			tempArray[bi] += tempArray[ai];
		}
		offset <<= 1;
	}

	if (threadId == 0) {
		tempArray[tam - 1] = 0;
	}

	//PARTE 2 REVERSE
	//Vamos recorriendo los datos del histograma para sumar las sumas parciales
	for (i = 1; i < tam; i <<= 1) {

		offset >>= 1;
		__syncthreads();
		if (threadId < i)
		{
			ai = offset*(2 * threadId + 1) - 1;
			bi = offset*(2 * threadId + 2) - 1;
			temp = tempArray[ai];
			tempArray[ai] = tempArray[bi];
			tempArray[bi] += temp;

		}
	}
	__syncthreads();

	//Guardamos los valores finales de vuelta al histograma
	d_histograma[id] = tempArray[threadId];
	d_histograma[id + tam / 2] = tempArray[threadId + tam / 2];

}

//KERNEL HISTOGRAMA
__global__
void histograma(unsigned int * d_histograma, const float* d_logLuminance, const float  min_logLum, const float  d_rango, const int tam, const int numbins) {
	// Seleccionamos el thread correspondiente
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	//Si nos salimos de la imagen, volvemos sin hacer nada
	if (i > tam)
		return;

	//Calculamos el bin correspondiente segun la formula facilitada
	int bin = ((d_logLuminance[i] - min_logLum) / d_rango) * numbins;
	//Y sumamos uno mediante atomicAdd
	atomicAdd(&d_histograma[bin], 1);


}

__global__
void calcularMinimo(float* d_Lum, float* d_LumMin, const float tam) {
	//Memoria compartida
	extern __shared__ float datos[];
	// Seleccionamos el thread correspondiente
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

	//Si nos salimos de la imagen, volvemos sin hacer nada
	if (i >= tam)
		return;

	//Copiamos los datos de entrada en la memoria compartida
	datos[tid] = d_Lum[i];
	__syncthreads();


	//Recorremos la mitad de los datos, cogiendo el minimo entre el dato actual y el dato actual + la mitad
	//Asi iremos cogiendo en cada pase del for dos elementos, para ir dejando el minimo de ellos
	for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
		if (tid < s) {
			datos[tid] = min(datos[tid], datos[tid + s]);
		}
		__syncthreads();
	}
	//Copiamos el valor final al dato de salida
	if (tid == 0) {
		d_LumMin[blockIdx.x] = datos[0];
	}

}
__global__
void calcularMaximo(float* d_Lum, float* d_LumMax, const float tam) {
	//Memoria compartida
	extern __shared__ float datos[];
	// Seleccionamos el thread correspondiente
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

	//Si nos salimos de la imagen, volvemos sin hacer nada
	if (i >= tam)
		return;

	//Copiamos los datos de entrada en la memoria compartida
	datos[tid] = d_Lum[i];
	__syncthreads();

	//Recorremos la mitad de los datos, cogiendo el maximo entre el dato actual y el dato actual + la mitad
	//Asi iremos cogiendo en cada pase del for dos maximo, para ir dejando el minimo de ellos
	for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
		if (tid < s) {
			datos[tid] = max(datos[tid], datos[tid + s]);
		}
		__syncthreads();
	}

	//Copiamos el valor final al dato de salida
	if (tid == 0) {
		d_LumMax[blockIdx.x] = datos[0];
	}

}


void calculate_cdf(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	float &min_logLum,
	float &max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	/* TODO
	1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance
	2) Obtener el rango a representar
	3) Generar un histograma de todos los valores del canal logLuminance usando la formula
	bin = (Lum [i] - lumMin) / lumRange * numBins
	4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	de los valores de luminancia. Se debe almacenar en el puntero c_cdf
	*/
	//------------------------------------PARTE 1------------------------------------
	//---------------------------ENCONTRAR VALOR MAX Y MIN---------------------------

	//Definimos el tamaño del bloque y de la imagen
	int tam_block = numBins;
	int tam = numCols*numCols;

	//Creamos el blockSize y gridSize para las llamadas a los kernel
	const dim3 blockSize(tam_block);
	const dim3 gridSize((tam - 1) / tam_block + 1);

	//Variables necesarias para coger los valores maximo y minimos
	float * d_LumIMin;
	float * d_LumIMax;
	float * d_LumMin;
	float * d_LumMax;

	//Reservamos memoria y copiamos los datos necesarios en las variables anteriores
	checkCudaErrors(cudaMalloc(&d_LumMin, sizeof(float)*numRows*numCols));
	checkCudaErrors(cudaMalloc(&d_LumMax, sizeof(float)*numRows*numCols));
	checkCudaErrors(cudaMalloc(&d_LumIMin, sizeof(float)*numRows*numCols));
	checkCudaErrors(cudaMalloc(&d_LumIMax, sizeof(float)*numRows*numCols));
	checkCudaErrors(cudaMemcpy(d_LumIMin, d_logLuminance, sizeof(float)*numRows*numCols, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_LumIMax, d_logLuminance, sizeof(float)*numRows*numCols, cudaMemcpyDeviceToDevice));

	//Como es necesario realziar varios pasos, creamos un bucle FOR que pare cuando quede un elemento
	while (tam >1) {

		//Se calcula el nuevo GridSize que se va reduciendo cada iteracion
		dim3 iGrid((tam - 1) / tam_block + 1);

		//Ejecutamos el kernel CalcularMinimo
		calcularMinimo << <iGrid, blockSize, blockSize.x * sizeof(float) >> > (d_LumIMin, d_LumMin, numRows*numCols);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		//Ejecutamos el kernel CalcularMaximo
		calcularMaximo << <iGrid, blockSize, blockSize.x * sizeof(float) >> > (d_LumIMax, d_LumMax, numRows*numCols);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		//Cambiamos los valores para que en las siguientes iteraciones tengan los valores actualizados
		d_LumIMin = d_LumMin;
		d_LumIMax = d_LumMax;

		//Y reducimos el tamaño para realizar la comprobación en el WHILE
		tam = (tam - 1) / (tam_block + 1);

	}

	//Una vez terminadas las iteraciones, copiamos los valores maximo y minimo a los parametros correspondientes
	checkCudaErrors(cudaMemcpy(&min_logLum, d_LumMin, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_LumMax, sizeof(float), cudaMemcpyDeviceToHost));
	printf("El minimo es: %f\n", min_logLum);
	printf("El maximo es: %f\n", max_logLum);

	//Y liberamos memoria
	checkCudaErrors(cudaFree(d_LumMax));
	checkCudaErrors(cudaFree(d_LumMin));
	//checkCudaErrors(cudaFree(d_LumIMax));
	//checkCudaErrors(cudaFree(d_LumIMin));


	//------------------------------------PARTE 2------------------------------------
	//---------------------------------OBTENER RANGO---------------------------------

	//Restamos el valor maximo menos el minimo para obtener el rango de valores
	float d_rango = max_logLum - min_logLum;
	printf("El rango es: %f\n", d_rango);


	//------------------------------------PARTE 3------------------------------------
	//----------------------------------HISTOGRAMA-----------------------------------

	//Creamos la variable del histograma
	unsigned int * d_histograma;
	//Creamos los valores de GridSize y BlockSize correspondientes para este kernel
	const dim3 blockSizeHis(numBins);
	const dim3 gridSizeHis(((numCols*numRows) - 1) / numBins + 1);

	printf("El numero de bins es: %d\n", numBins);

	//Reservamos la memoria y ponemos los valores a 0 en el histograma
	checkCudaErrors(cudaMalloc(&d_histograma, numBins * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_histograma, 0, numBins * sizeof(unsigned int)));

	//Ejecutamos el kernel histograma para obtener sus valores
	histograma << <gridSizeHis, blockSizeHis >> > (d_histograma, d_logLuminance, min_logLum, d_rango, numCols*numRows, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


	//------------------------------------PARTE 4------------------------------------
	//-------------------------------------SCAN-----------------------------------

	//Creamos los valores de GridSize y BlockSize correspondientes para este kernel
	const dim3 blockSizeScan(numBins);
	const dim3 gridSizeScan((numBins - 1) / numBins + 1);

	//Ejecutamos el kernel Scan para obtener el valor definitivo de d_cdf
	scan << <gridSizeScan, blockSizeScan >> > (d_histograma, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//Copiamos el valor resultante del kernel en d_cdf y liberamos memoria
	checkCudaErrors(cudaMemcpy(d_cdf, d_histograma, sizeof(unsigned int)*numBins, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaFree(d_histograma));

}