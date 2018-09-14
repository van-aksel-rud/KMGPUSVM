#include "UtilityFunctions.h"
#include <helper_functions.h>
#include <helper_cuda.h>


void printKernelBlock(size_t block_size, float *devicePtr) {
	float *blk = new float[block_size*block_size];

	checkCudaErrors(cudaMemcpy(blk, devicePtr, block_size*block_size * sizeof(float), cudaMemcpyDeviceToHost));
	for (size_t i = 0; i < block_size; i++) {
		std::cout << std::endl;
		for (size_t j = 0; j < block_size; j++) {
			std::cout << std::setw(10);
			std::cout << blk[i*block_size + j];
		}
	}
	std::cout << std::endl;

	delete[] blk;

}

void printDeviceMemory(size_t kernelRows, size_t colWidth, float *d_Rand) {
	std::vector<float> rnd;
	rnd.resize(kernelRows);
	checkCudaErrors(cudaMemcpy(&rnd[0], d_Rand, kernelRows * sizeof(float), cudaMemcpyDeviceToHost));
	size_t k = 0;
	while (k < rnd.size()) {
		for (size_t col = 0; col < colWidth; col++) {
			if (k < rnd.size())
				std::cout << std::setw(14) << rnd[k++];
		}
		std::cout << std::endl;
	}
}

float printDeviceMemoryTotal(size_t kernelRows, float *d_Rand) {
	std::vector<float> rnd;
	rnd.resize(kernelRows);
	checkCudaErrors(cudaMemcpy(&rnd[0], d_Rand, kernelRows * sizeof(float), cudaMemcpyDeviceToHost));
	size_t k = 0;
	float ss = 0;
	while (k < rnd.size()) {
		ss += rnd[k++];
	}
	std::cout << std::endl << ss;

	return ss;
}

template<size_t BLOCK_SIZE> void svm(const std::vector<float>& alpha, const std::vector<float>& y, const std::vector<KERNEL_BLOCK<BLOCK_SIZE>>& kernel, std::vector<float>& pred) {
	for (size_t i = 0; i < kernel.size(); i++) {
		const KERNEL_BLOCK<BLOCK_SIZE>& pBlock = kernel[i];
		size_t row = pBlock.i*BLOCK_SIZE;
		size_t col = pBlock.j*BLOCK_SIZE;
		for (size_t k = 0; k < BLOCK_SIZE; k++) {
			for (size_t m = 0; m < BLOCK_SIZE; m++) {
				pred[col + m] += y[row + k] * alpha[row + k] * pBlock.pData[k][m];
			}
		}
	}
}


void writeDeviceData(float *d_Rand, size_t kernelRows, FILE *pFile) {
	std::vector<float> rnd;
	rnd.resize(kernelRows);
	checkCudaErrors(cudaMemcpy(&rnd[0], d_Rand, kernelRows * sizeof(float), cudaMemcpyDeviceToHost));
	fwrite(&rnd[0], sizeof(float), kernelRows, pFile);
}