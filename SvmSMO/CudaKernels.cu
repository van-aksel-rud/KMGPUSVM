#include "CudaKernels.h"
#include "KernelBlock.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_string.h>

__device__  void get(SparseVector* from, int i, SparseVector** ret) {
	SparseVector *pos = from;
	int j = 0;
	for (; j < i; j++) {
		if (pos->m_nafter == 0)
			break;
		*(char**)&pos += int(pos->m_size);
	}
	*ret = (j == i) ? pos : 0;
}


//Computing kernels in global memory
__global__ void SUB_CHILD_RBF(const SparseVector *left, const SparseVector* right, KernelParameters param, double *res) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const float *pLeft = &left->m_data[bx*BLOCK_SIZE*int(left->m_cols)];
	const float *pRight = &right->m_data[by*BLOCK_SIZE*int(right->m_cols)];
	double dNorm = left->m_rows * right->m_rows;
	int nc = left->m_cols;

	if ((bx*BLOCK_SIZE + tx < left->m_rows) && (by*BLOCK_SIZE + ty < right->m_rows)) {
		double d = 0;
		for (size_t i = 0; i < nc; i++) {
			float fl = pLeft[tx*nc + i];
			float fr = pRight[ty*nc + i];
			double dd = fl - fr;
			d += dd*dd;
		}
		atomicAdd(res, exp(-param.m_gamma*d) / dNorm);
	}

}

__global__ void CHILD_RBF(int C, KernelBlock *pBlock, KernelParameters param, bool same) {

	int row = blockIdx.x;
	int col = same ? row : blockIdx.y;

	SparseVector* left = 0;
	SparseVector* right = 0;
	get(pBlock->rows, row, &left);
	get(pBlock->cols, col, &right);
	double *res = &((double*)&pBlock->pData[0][0])[same ? row : (row*C + col)];
	*res = 0;
	if ((left == 0) || (right == 0))
		return;


	int row_iter = (left->m_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int col_iter = (right->m_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

	SUB_CHILD_RBF << < dim3(row_iter, col_iter), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (left, right, param, res);
}

__global__ void RBF_CUDA(int R, int C, KernelBlock *pBlocks, KernelParameters param, bool same)
{
	int bx = blockIdx.x;

	KernelBlock *pBlock = &pBlocks[bx];

	CHILD_RBF << <dim3(R, C), 1 >> > (C, pBlock, param, same);
}
