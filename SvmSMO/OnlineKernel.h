#pragma once

#include "BatchV.h"
#include <unordered_map>
#include <vector>
#include <set>

#include "KernelBlock.h"
#include <numeric>
#include <stdio.h>


__global__ void RBF_CUDA(int R, int C, KernelBlock *pBlocks, KernelParameters param, bool same);

template<class T> class POSLEN {

	T pos;
	size_t len;
public:
	POSLEN(T p, size_t n) :
		pos(p),
		len(n) {

	}
	operator size_t() const { return len; }
	operator T () const { return pos;  }
	T operator [](size_t i) const {
		if (i >= len) {
			throw "Invalid index";
		}
		else
			return pos[i];
	}
};

class OnlineKernel {
public:
	class Row {
		friend class OnlineKernel;
		int m_rowIndex;
		OnlineKernel& K;
	public:
		Row(OnlineKernel& k, int row) : 
			m_rowIndex(row),
			K(k)
		{
		}

		double operator[](int index) {

			if ((K.m_nCacheSize != 1) && (m_rowIndex == index))
				return K.getDiag(index);

			std::unordered_map<int, RowInfo*>::iterator itf = K.m_mapPos.find(m_rowIndex);
			if (itf != K.m_mapPos.end()) {

				K.m_nCacheHits++;
				RowInfo *pos = itf->second;
				if (pos->prev) {
					if (pos->next == 0)
						K.m_pLast = pos->prev;
					pos->prev->next = pos->next;
					if (pos->next)
						pos->next->prev = pos->prev;
					pos->prev = 0;
					pos->next = K.m_pTop;
					K.m_pTop->prev = pos;
					K.m_pTop = pos;
				}

				return K.m_ppK[itf->second->id][index];
			}
			else
			{
				K.m_nCacheMiss++;
				double *pRow = K.readRow(m_rowIndex);
				return pRow[index];
			}
		}
	};

private:
	BatchV **m_pBatch;
	SparseVector *m_pVecBuf;
	SparseVector *m_pVecD;
	SparseVector *m_devVecBuf;
	SparseVector *m_devVecD;
	KernelBlock* m_pBlocks;
	int m_nTotalRows;
	int m_nMaxBatchRows;
	int m_nMaxBatchSize;
	int m_NumBatches;
	double **m_ppK;
	char *m_pY;
	std::vector<double> m_vecAlpha;
	int m_nCacheSize;

	KernelParameters m_param;

	std::vector<char> m_YPrecomputed;
	double **m_ppKPrecomputed;


	struct RowInfo {
		RowInfo(int i = 0, int n = 0) :
			id(i),
			index(n),
			prev(0),
			next(0)
		{
		}
		RowInfo *prev;
		RowInfo *next;
		int id;
		int index;
		bool operator ==(const RowInfo& r) const {
			return index == r.index;
		}

	};

	RowInfo* m_pLast;
	RowInfo* m_pTop;
	RowInfo* m_pCurr;
	std::unordered_map<int, RowInfo*> m_mapPos;
	size_t m_nCacheHits;
	size_t m_nCacheMiss;

	std::vector<double> m_vecDiag;

	double getDiag(int index) {
		return m_vecDiag[index];
	}

	double* readRow(int index) {

		if (m_mapPos.size() < m_nCacheSize) {
			m_pCurr->index = index;
			int id = m_pCurr->id;
			m_mapPos[index] = m_pCurr;
			m_pCurr = m_pCurr->next;

			SparseVector *pD = getVector(index, true);
			compute(m_ppK[id], pD);
			return m_ppK[id];
		}
		else {
			std::unordered_map<int, RowInfo*>::iterator itf = m_mapPos.find(m_pLast->index);
			m_mapPos.erase(itf);
			m_mapPos[index] = m_pLast;
			m_pLast->index = index;
			int id = m_pLast->id;

			RowInfo *pos = m_pLast;
			m_pLast = m_pLast->prev;
			m_pLast->next = 0;
			pos->next = m_pTop;
			m_pTop->prev = pos;
			m_pTop = pos;
			m_pTop->prev = 0;

			SparseVector *pD = getVector(index, true);
			compute(m_ppK[id], pD);
			return m_ppK[id];
		}
	}

	std::vector<double> computeRow(SparseVector *pD) {
		std::vector<double> ret;
		ret.resize(m_nTotalRows);
		compute(&ret[0], pD);
		return ret;
	}

	void getBatch(int index, int& b, int& i) {
		int nSoFar = m_pBatch[0]->getNumRows();
		int k = 1;
		while (index >= nSoFar)
			nSoFar += m_pBatch[k++]->getNumRows();
		b = --k;
		i = index - nSoFar + m_pBatch[k++]->getNumRows();
	}

	SparseVector *getVector(int index, bool pivot = false) {

		int b = 0;
		int idiag = 0;
		getBatch(index, b, idiag);

		SparseVector *buf = pivot ? m_pVecD : m_pVecBuf;
		m_pBatch[b]->read(buf);

		return (*buf)[idiag];
	}


	void compute(double *D, SparseVector *pD = 0) {
		int writePos = 0;
		for (size_t i = 0; i < m_NumBatches; i++) {

			m_pBatch[i]->read(m_pVecBuf);
			int nr = m_pBatch[i]->getNumRows();
			int nIter = (nr + BLOCK_SIZE*BLOCK_SIZE - 1) / (BLOCK_SIZE*BLOCK_SIZE);
			checkCudaErrors(cudaMemcpy(m_devVecBuf, m_pVecBuf, m_pBatch[i]->getTotalSize(), cudaMemcpyHostToDevice));
			for (size_t j = 0; j < nIter; j++) {
				SparseVector *pos = (*m_pVecBuf)[j*BLOCK_SIZE*BLOCK_SIZE];
				int offset = (char*)pos - (char*)m_pVecBuf;
				char *pDev = (char*)m_devVecBuf + offset;
				checkCudaErrors(cudaMemcpy(&m_pBlocks[j].rows, &pDev, sizeof(SparseVector*), cudaMemcpyHostToDevice));
				if (pD == 0) {
					checkCudaErrors(cudaMemcpy(&m_pBlocks[j].cols, &pDev, sizeof(SparseVector*), cudaMemcpyHostToDevice));
				}
				else {
					checkCudaErrors(cudaMemcpy(m_devVecD, pD, pD->m_size, cudaMemcpyHostToDevice));
					checkCudaErrors(cudaMemcpy(&m_pBlocks[j].cols, &m_devVecD, sizeof(SparseVector*), cudaMemcpyHostToDevice));

				}
			}

			RBF_CUDA << <nIter, 1 >> > (BLOCK_SIZE*BLOCK_SIZE, 1, m_pBlocks, m_param, pD == 0 ? true : false);
			checkCudaErrors(cudaDeviceSynchronize());

			size_t j = 0;
			int n = m_pBatch[i]->getNumRows() / (BLOCK_SIZE*BLOCK_SIZE);
			for (; j < n; j++) {
				checkCudaErrors(cudaMemcpy(&D[writePos + j*BLOCK_SIZE*BLOCK_SIZE], m_pBlocks[j].pData, BLOCK_SIZE*BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
			}
			int rem = m_pBatch[i]->getNumRows() % (BLOCK_SIZE*BLOCK_SIZE);
			if (rem != 0)
				checkCudaErrors(cudaMemcpy(&D[writePos + j*BLOCK_SIZE*BLOCK_SIZE], m_pBlocks[j].pData, rem * sizeof(double), cudaMemcpyDeviceToHost));
			writePos += m_pBatch[i]->getNumRows();
		}
	}

public:
	OnlineKernel(const KernelParameters& param, const char* strBatch, int N, int cacheSize = 1000) :
		m_param(param),
		m_pBatch(0),
		m_NumBatches(N),
		m_nCacheSize(cacheSize),
		m_nTotalRows(0),
		m_pBlocks(0),
		m_pVecD(0),
		m_pVecBuf(0),
		m_devVecD(0),
		m_devVecBuf(0),
		m_pY(0),
		m_ppKPrecomputed(0),
		m_nCacheHits(0),
		m_nCacheMiss(0),
		m_nMaxBatchRows(0),
		m_nMaxBatchSize(0)
	{
		m_NumBatches = (m_NumBatches == 0) ? BatchV::getMaxBatchId(strBatch)+1 : m_NumBatches;

		m_pBatch = new BatchV*[m_NumBatches];
		for (size_t i = 0; i < m_NumBatches; i++) {
			m_pBatch[i] = new BatchV(strBatch, i);
			m_nTotalRows += m_pBatch[i]->getNumRows();
			m_nMaxBatchRows = (m_pBatch[i]->getNumRows() > m_nMaxBatchRows) ? m_pBatch[i]->getNumRows() : m_nMaxBatchRows;
			m_nMaxBatchSize = (m_pBatch[i]->getTotalSize() > m_nMaxBatchSize) ? m_pBatch[i]->getTotalSize() : m_nMaxBatchSize;
		}

		m_pY = new char[m_nTotalRows];
		m_pY[0] = 0;
		m_ppK = new double*[cacheSize];
		for (size_t i = 0; i < m_nCacheSize; i++) {
			m_ppK[i] = new double[m_nTotalRows];
		}


		m_pVecD = (SparseVector*) new char[m_nMaxBatchSize];
		checkCudaErrors(cudaMalloc(&m_devVecD, m_nMaxBatchSize));
		m_pVecBuf = (SparseVector*) new char[m_nMaxBatchSize];
		checkCudaErrors(cudaMalloc(&m_devVecBuf, m_nMaxBatchSize));
		int n = (m_nMaxBatchRows + BLOCK_SIZE*BLOCK_SIZE - 1) / (BLOCK_SIZE*BLOCK_SIZE);
		checkCudaErrors(cudaMalloc(&m_pBlocks, n*sizeof(KernelBlock)));
		for (int j = 0; j < n; j++) {
			checkCudaErrors(cudaMemcpy(&m_pBlocks[j].rows, &m_devVecBuf, sizeof(char*), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(&m_pBlocks[j].cols, &m_devVecD, sizeof(char*), cudaMemcpyHostToDevice));
		}

		m_pTop = new RowInfo();
		m_pLast = m_pTop;
		m_pCurr = m_pTop;
		for (int i = 1; i < m_nCacheSize; i++) {
			m_pLast->next = new RowInfo(i);
			m_pLast->next->prev = m_pLast;
			m_pLast = m_pLast->next;
 		}

		if (m_nCacheSize != 1) {
			m_vecDiag.resize(m_nTotalRows);
			compute(&m_vecDiag[0]);
		}
		else {
			m_param = m_pBatch[0]->getParameters();
		}

	}
public:
	OnlineKernel(const char* strBatch) :
		OnlineKernel(KernelParameters(), strBatch, 1, 1)
	{

	}
	OnlineKernel(int target, const char* strBatch) :
		OnlineKernel(KernelParameters(target), strBatch, 1,1)
	{

	}

	~OnlineKernel() {
		delete[] m_pY;
		for (size_t i = 0; i < m_nCacheSize; i++) {
			delete[] m_ppK[i];
		}
		delete[] m_ppK;
		for (size_t i = 0; i < m_NumBatches; i++) {
			delete m_pBatch[i];
		}
		delete[] m_pBatch;
		if (m_pBlocks)
			cudaFree(m_pBlocks);
		if (m_pVecD) {
			delete[](char*)m_pVecD;
			delete[](char*)m_pVecBuf;
			cudaFree(m_devVecD);
			cudaFree(m_devVecBuf);
		}
	}

	double sqr(double *l, double *r, int N) {
		double d = 0;
		for (int i = 0; i < N; i++)
			d += pow(l[i] - r[i], 2);
		return d;
	}

	Row operator[](int index) {

		return Row(*this, index);

	}

	int getNumRows() const {
		return m_nTotalRows;
	}

	char* getY() {
		int k = 0;
		if (m_pY[0] == 0) {
			for (size_t i = 0; i < m_NumBatches; i++) {
				for (int j = 0; j < m_pBatch[i]->getNumRows(); j++)
					m_pY[k++] = (char)(m_pBatch[i]->getMultiLabel(j) == m_param.m_nTarget ? 1 : -1);
			}
		}
		return m_pY;
	}

	std::vector<double>& getAlpha() {
		int k = 0;
		if (m_vecAlpha.empty()) {
			m_vecAlpha.resize(m_nTotalRows);
			for (size_t i = 0; i < m_NumBatches; i++) {
				for (int j = 0; j < m_pBatch[i]->getNumRows(); j++)
					m_vecAlpha[k++] = m_pBatch[i]->getAlpha(j);
			}
		}
		return m_vecAlpha;
	}

	double kval(SparseVector *left, SparseVector *right) {

		double k = 0;
		int nc = left->m_cols;
		double dNorm = left->m_rows*right->m_rows;

		for (size_t i = 0; i < left->m_rows; i++) {
			for (size_t j = 0; j < right->m_rows; j++) {
				double d = 0;
				for (size_t l = 0; l < left->m_cols; l++) {
					d += m_param.W[l] * pow(left->m_data[i*nc + l] - right->m_data[j*nc + l], 2);
				}
				k += exp(-m_param.m_gamma*d) / dNorm;
			}
		}

		return k;
	}

	double predict(SparseVector *pD) {
		if (m_nTotalRows == 0)
			return 0;
		std::vector<double> row;
		row.resize(m_nTotalRows);
		compute(&row[0], pD);

		char* Y = getY();
		std::vector<double>& A = getAlpha();
		double kv = 0;
		for (int i = 0; i < row.size(); i++) {
			kv += double(Y[i]) * A[i] * row[i];
		}
		kv -= m_param.m_rho;
		return kv;
	}

	void save(const char *szName) {
		FILE *pFile = fopen(szName, "wb");
		if (pFile) {
			fwrite(&m_nTotalRows, sizeof(int), 1, pFile);
			std::vector<double> row;
			row.assign(m_nTotalRows+1, 0);
			for (int i = 0; i < m_nTotalRows; i++) {
				SparseVector *pD = getVector(i, true);
				row[0] = pD->m_y;
				compute(&row[1], pD);
				fwrite(&row[0], sizeof(double), m_nTotalRows+1, pFile);
			}
			fclose(pFile);
		}
	}


	void saveModel(const char *szName, const std::vector<double>& alpha, const KernelParameters& param) {

		std::ostringstream ossModelName;
		ossModelName << szName << "_" << param.m_nTarget;
		std::string modelFile = ossModelName.str();
		modelFile += "_0.dat";
		std::remove(modelFile.c_str());

		BatchV batchSV(ossModelName.str().c_str(), 0, param);
;
		for (size_t i = 0; i < alpha.size(); i++) {
			if (alpha[i] != 0) {
				int b = 0;
				int bi = 0;
				getBatch(i, b, bi);
				SparseVector *pSv = m_pBatch[b]->getData(bi);
				pSv->alpha = alpha[i];
				batchSV.addRow(*pSv);
				delete[](char*)pSv;
			}
		}
	}

	bool verify() {
		double err = 0;
		for (int i = 0; i < 10; i++) {
			int idx = rand() % m_nTotalRows;

			SparseVector *pD = getVector(idx, true);
			compute(m_ppK[0], pD);

			int idy = rand() % m_nTotalRows;
			SparseVector *pV = getVector(idy);

			double k = kval(pD, pV);
			err += pow(k - m_ppK[0][idy], 2);
			k = kval(pD, pD);
			err += pow(k - m_ppK[0][idx], 2);

		}
		return err < 1e-6;
	}

	void readPrecomputed(const char *szName, std::vector<char>& y, double ***ppK) {

		FILE *pKernel = fopen(szName, "rb");

		if (pKernel) {
			int N = 0;
			fread(&N, sizeof(int), 1, pKernel);

			*ppK = new double*[N];
			for (int i = 0; i < N; i++)
				(*ppK)[i] = new double[N];

			double **m_pK = *ppK;


			y.resize(N);

			for (size_t i = 0; i < N; i++) {
				double dy = 0;
				fread(&dy, sizeof(double), 1, pKernel);
				y[i] = dy;
				fread(m_pK[i], sizeof(double), N, pKernel);
			}

			fclose(pKernel);
		}
	}

	size_t getCacheSize() const {
		return m_mapPos.size();
	}

	double getCacheHitRatio() {
		double r = double(m_nCacheHits) / double(m_nCacheHits + m_nCacheMiss + 0.001);
		m_nCacheHits = 0;
		m_nCacheMiss = 0;
		return r;
	}

	double getCacheMissRatio() {
		double r = double(m_nCacheMiss) / double(m_nCacheHits + m_nCacheMiss + 0.001);
		m_nCacheHits = 0;
		m_nCacheMiss = 0;
		return r;
	}

	double getCacheMissCount() {
		double r = m_nCacheMiss;
		m_nCacheHits = 0;
		m_nCacheMiss = 0;
		return r;
	}

	KernelParameters getParam() const {
		return m_param;
	}

};