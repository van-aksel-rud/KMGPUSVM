#pragma once

//#define _CRTDBG_MAP_ALLOC  
//#include <stdlib.h>  
//#include <crtdbg.h> 

#define BLOCK_SIZE 16
#define BATCH_SIZE 1000
#define COLUMN_BLOCK 4

#pragma pack(1)
struct KernelParameters {
	KernelParameters(int target = 0) :
		m_rho(0),
		m_gamma(0),
		m_nTarget(target)
	{
		W[0] = W[1] = W[2] = 1;
	}
	void print() const {
		std::cout << std::endl << "------------------------------------------------------------------------------------------------------------------";
		std::cout << std::endl << "Rho   " << m_rho;
		std::cout << std::endl << "Gamma " << m_gamma;
		std::cout << std::endl << "Class " << m_nTarget;
		std::cout << std::endl << "W     " << W[0] << " " << W[1] << " " << W[2];
		std::cout << std::endl << "------------------------------------------------------------------------------------------------------------------";
	}
	double m_rho;
	double m_gamma;
	__int32 m_nTarget;
	double W[3];
};
#pragma pack()


struct SparseVector {
	__int32 m_nafter;
	__int32 m_size;
	__int32 m_rows;
	__int32 m_cols;
	float alpha;
	float m_y;
	float m_data[1];
	SparseVector() :
		m_size(0),
		m_y(0),
		alpha(0),
		m_rows(0),
		m_cols(0)
	{
		m_data[0] = 0;
	}
	SparseVector* operator[] (int i) {
		SparseVector *pos = this;
		int j = 0;
		for (; j < i; j++) {
			if (pos->m_nafter == 0)
				break;
			*(char**)&pos += int(pos->m_size);
		}
		return (j == i) ? pos : 0;
	}
};

