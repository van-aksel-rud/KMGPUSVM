#pragma once

#include "SparseVector.h"
#include <vector>


class KernelBlock {
public:
	SparseVector* rows;
	SparseVector* cols;
	double pData[BLOCK_SIZE][BLOCK_SIZE];
};
