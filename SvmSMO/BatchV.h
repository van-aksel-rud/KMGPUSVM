#pragma once

#include "SparseVector.h"
#include <vector>

#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <sstream>
#include <cstdio>

#pragma pack(1)
struct BatchHeader {
	BatchHeader(const KernelParameters& p) :
		param(p),
		m_numRec(0),
		m_nTotalSize(0)
	{}
	__int32 m_numRec;
	__int32 m_nTotalSize;
	KernelParameters param;
};
#pragma pack()

class BatchV {
	int *m_pOffsets;
	FILE *m_pFile;
	std::string m_strName;
	BatchHeader m_header;
	int m_id;

public:
	BatchV(const std::string& strName, size_t id) :
		BatchV(strName, id, KernelParameters())
	{
	}

	BatchV(const std::string& strName, size_t id, const KernelParameters& p) :
		m_strName(strName),
		m_pOffsets(0),
		m_pFile(0),
		m_id(id),
		m_header(p)
	{
		std::ostringstream strFilename;
		strFilename << strName << "_" << id << ".dat";
		m_strName = strFilename.str();
		if ((m_pFile = fopen(m_strName.c_str(), "a+b")) != 0) {
			fseek(m_pFile, 0, SEEK_END);
			int nFileSize = ftell(m_pFile);
			if (nFileSize == 0) {
				fwrite(&m_header, sizeof(BatchHeader), 1, m_pFile);
				m_pOffsets = new int;
				m_pOffsets[0] = ftell(m_pFile);
				return;
			}
			fseek(m_pFile, 0, SEEK_SET);
			fread(&m_header, sizeof(BatchHeader), 1, m_pFile);
			m_pOffsets = new int[(int)m_header.m_numRec];
			m_pOffsets[0] = ftell(m_pFile);
			for (size_t recNum = 0; recNum < int(m_header.m_numRec -1); recNum++) {
				__int32 f[2] = { 0 };
				fread(&f, sizeof(__int32), 2, m_pFile);
				fseek(m_pFile, int(f[1]) - 2* sizeof(float), SEEK_CUR);
				m_pOffsets[recNum + 1] = ftell(m_pFile);
			}
		}
	}

	int getId() const {
		return m_id;
	}

	~BatchV() {
		if (m_pFile) {
			fseek(m_pFile, 0, SEEK_END);
			int nTotalSize = ftell(m_pFile);
			fclose(m_pFile);
			if (m_header.m_nTotalSize != nTotalSize) {
				FILE *pFile = fopen(m_strName.c_str(), "r+b");
				fseek(pFile, 0, SEEK_SET);
				m_header.m_nTotalSize = nTotalSize;
				fwrite(&m_header, sizeof(BatchHeader), 1, m_pFile);

				__int32 numAfter = m_header.m_numRec - 1;
				for (size_t recNum = 0; recNum < m_header.m_numRec; recNum++) {
					fseek(pFile, m_pOffsets[recNum], SEEK_SET);
					fwrite(&numAfter, sizeof(__int32), 1, pFile);
					numAfter--;
				}
				fclose(pFile);
			}
			delete[] m_pOffsets;
		}
	}


	int getMultiLabel(int i) {
		if (i > m_header.m_numRec)
			return 0;
		fseek(m_pFile, m_pOffsets[i] + 5 * sizeof(float), 0);
		float f_y = 0;
		fread(&f_y, sizeof(float), 1, m_pFile);
		return int(f_y);
	}

	float getAlpha(int i) {
		if (i > m_header.m_numRec)
			return 0;
		fseek(m_pFile, m_pOffsets[i] + 4 * sizeof(float), 0);
		float f_a = 0;
		fread(&f_a, sizeof(float), 1, m_pFile);
		return f_a;
	}

	SparseVector* getData(int i) {
		if (i > m_header.m_numRec)
			return 0;
		fseek(m_pFile, m_pOffsets[i], SEEK_SET);
		__int32 f[2] = { 0 };
		fread(f, sizeof(__int32), 2, m_pFile);
		SparseVector *pRet = (SparseVector*)new char[f[1]];
		pRet->m_nafter = f[0];
		pRet->m_size = f[1];
		fread(&pRet->m_rows, sizeof(char), int(f[1]) - 2*sizeof(__int32), m_pFile);
		return pRet;
	}

	int getTotalSize() const {
		return m_header.m_nTotalSize;
	}

	void read(SparseVector* pVec) {
		fseek(m_pFile, m_pOffsets[0], SEEK_SET);
		fread(pVec, m_header.m_nTotalSize, 1, m_pFile);
		while (pVec->m_nafter) {
			pVec->m_y = int(pVec->m_y == m_header.param.m_nTarget) ? 1 : -1;
			pVec = (SparseVector*)((char*)pVec + pVec->m_size);
		}
	}

	int read(SparseVector** pVec) {
		*pVec = (SparseVector*) new char[m_header.m_nTotalSize];
		read(*pVec);
		return m_header.m_nTotalSize;
	}

	KernelParameters getParameters() const {
		return m_header.param;
	}

	size_t getNumRows() const {
		return m_header.m_numRec;
	}

	void addRow(const SparseVector& rv) {

		int *pOffsets = new int[m_header.m_numRec + 1];
		memcpy(pOffsets, m_pOffsets, m_header.m_numRec*sizeof(int));
		delete[] m_pOffsets;
		m_pOffsets = pOffsets;
		fseek(m_pFile, 0, SEEK_END);
		m_pOffsets[m_header.m_numRec] = ftell(m_pFile);
		m_header.m_numRec += 1;
		fwrite(&rv, int(rv.m_size), 1, m_pFile);
	}

	static int getMaxBatchId(const std::string& strName) {
		int id = 0;
		for (;; id++) {
			std::ostringstream strFilename;
			strFilename << strName << "_" << id << ".dat";
			std::string strName = strFilename.str();
			FILE *pFile = 0;
			if ((pFile = fopen(strName.c_str(), "r")) != 0) {
				fclose(pFile);
			}
			else
				break;
		}
		return id - 1;
	}
};
