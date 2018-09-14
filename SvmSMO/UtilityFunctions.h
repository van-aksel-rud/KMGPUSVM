#pragma once
#include <vector>


void printKernelBlock(size_t block_size, float *devicePtr);
void printDeviceMemory(size_t kernelRows, size_t colWidth, float *d_Rand);
float printDeviceMemoryTotal(size_t kernelRows, float *d_Rand);
template<size_t BLOCK_SIZE> void svm(const std::vector<float>& alpha, const std::vector<float>& y, const std::vector<KERNEL_BLOCK<BLOCK_SIZE>>& kernel, std::vector<float>& pred);
void writeDeviceData(float *d_Rand, size_t kernelRows, FILE *pFile);
