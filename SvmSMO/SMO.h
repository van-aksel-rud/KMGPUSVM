#pragma once
#include "OnlineKernel.h"
#include <chrono>


#ifndef min
template <class T> static inline T min(T x, T y) { return (x<y) ? x : y; }
#endif
#ifndef max
template <class T> static inline T max(T x, T y) { return (x>y) ? x : y; }
#endif

#define INF HUGE_VAL
#define TAU 1e-12


class SMO {
	std::vector<double> G;		// gradient of objective function
	std::vector<double> G_bar;		// gradient, if we treat free variables as 0
	std::vector<double> m_alpha;
	double m_rho;
	OnlineKernel& K;
	int m_lSize;

	int select_working_set(double C, int &out_i, int &out_j)
	{
		// return i,j such that
		// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
		// j: minimizes the decrease of obj value
		//    (if quadratic coefficeint <= 0, replace it with tau)
		//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

		double eps = 1e-6;
		double Gmax = -INF;
		double Gmax2 = -INF;
		int Gmax_idx = -1;
		int Gmin_idx = -1;
		double obj_diff_min = INF;

		int COLS = K.getNumRows();
		char* Y = K.getY();

		for (int t = 0; t < COLS; t++) {
			double thr = Y[t] == 1 ? C : 0;

			if ((m_alpha[t] != thr) && (-double(Y[t]) * G[t] >= Gmax))
			{
				Gmax = -double(Y[t]) * G[t];
				Gmax_idx = t;
			}
		}

		int i = Gmax_idx;

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		int prevScond = 0;
		for (int j = 0; j< COLS; j++)
		{

			std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

			unsigned int currSecond = time_span / 1000;
			if (prevScond != currSecond) {
				std::cout << std::endl << std::setw(10) << " " << " Cache Miss count " << K.getCacheMissCount();
				prevScond = currSecond;
			}

			if (Y[j] == +1)
			{
				if (m_alpha[j] != 0)
				{
					double grad_diff = Gmax + G[j];
					if (G[j] >= Gmax2)
						Gmax2 = G[j];
					if (grad_diff > 0)
					{
						double obj_diff;
						double quad_coef = K[i][i] + K[j][j] - 2.0*double(Y[j]) * K[i][j];
						if (quad_coef > 0)
							obj_diff = -(grad_diff*grad_diff) / quad_coef;
						else
							obj_diff = -(grad_diff*grad_diff) / TAU;

						if (obj_diff <= obj_diff_min)
						{
							Gmin_idx = j;
							obj_diff_min = obj_diff;
						}
					}
				}
			}
			else
			{
				if (m_alpha[j] != C)
				{
					double grad_diff = Gmax - G[j];
					if (-G[j] >= Gmax2)
						Gmax2 = -G[j];
					if (grad_diff > 0)
					{
						double obj_diff;
						double quad_coef = K[i][i] + K[j][j] + 2.0*double(Y[j]) * K[i][j];
						if (quad_coef > 0)
							obj_diff = -(grad_diff*grad_diff) / quad_coef;
						else
							obj_diff = -(grad_diff*grad_diff) / TAU;

						if (obj_diff <= obj_diff_min)
						{
							Gmin_idx = j;
							obj_diff_min = obj_diff;
						}
					}
				}
			}
		}

		if (Gmax + Gmax2 < eps || Gmin_idx == -1)
			return 1;

		out_i = Gmax_idx;
		out_j = Gmin_idx;
		return 0;
	}

	double calculate_rho(char* Y, double C)
	{
		int COLS = K.getNumRows();
		double r;
		int nr_free = 0;
		double ub = INF, lb = -INF, sum_free = 0;

		for (int i = 0; i< COLS; i++)
		{
			double yG = Y[i] * G[i];

			if (m_alpha[i] == C)
			{
				if (Y[i] == -1)
					ub = min(ub, yG);
				else
					lb = max(lb, yG);
			}
			else if (m_alpha[i] == 0)
			{
				if (Y[i] == +1)
					ub = min(ub, yG);
				else
					lb = max(lb, yG);
			}
			else
			{
				++nr_free;
				sum_free += yG;
			}
		}

		if (nr_free>0)
			r = sum_free / nr_free;
		else
			r = (ub + lb) / 2;

		return r;
	}
public:
	SMO(OnlineKernel& k, int L) :
		K(k),
		m_lSize(L),
		m_rho(0)
	{
		G.resize(L);
		G_bar.resize(L);
		m_alpha.resize(L);
	};
	virtual ~SMO() {};

	struct SolutionInfo {
		SolutionInfo(const std::vector<double>& a) :
			alpha(a),
			rho(0),
			numiter(0),
			nNegAlpha(0),
			nPosAlpha(0)
		{

		}
		double rho;
		const std::vector<double>& alpha;
		double obj;
		int numiter;
		int nNegAlpha;
		int nPosAlpha;
	};


	SMO::SolutionInfo SMO::Solve(double C) {
		SMO::SolutionInfo si(m_alpha);

		char* Y = K.getY();
		int COLS = K.getNumRows();

		std::cout << std::endl << "Optimizing " << COLS << " rows";


		// initialize gradient
		for (int i = 0; i < COLS; i++)
		{
			G[i] = -1;
			G_bar[i] = 0;
		}

		// optimization step

		int max_iter = max(10000000, COLS>INT_MAX / 100 ? INT_MAX : 100 * COLS);
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		int prevScond = 0;

		for (si.numiter = 0; si.numiter < max_iter; si.numiter++)
		{
			std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

			unsigned int currSecond = time_span / 1000;
			if (prevScond != currSecond) {
				std::cout << std::endl << "Iter " << std::setw(10) << si.numiter << "Cache Size " << std::setw(10) << K.getCacheSize() << " Cache Miss count " << K.getCacheMissCount();
				prevScond = currSecond;
			}


			int i = 0;
			int j = 0;
			if (select_working_set(C, i, j) != 0)
				break;

			double old_alpha_i = m_alpha[i];
			double old_alpha_j = m_alpha[j];

			double thr = Y[i] == Y[j] ? C : 0;
			double sig = Y[i] == Y[j] ? 1 : -1;


			double quad_coef = K[i][i] + K[j][j] + 2 * double(Y[i]) * double(Y[j]) * K[i][j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (sig*G[i] - G[j]) / quad_coef;
			double diff = m_alpha[i] + sig*m_alpha[j];
			m_alpha[i] -= sig*delta;
			m_alpha[j] += delta;


			if (sig*diff > sig*thr)
			{
				if (sig*m_alpha[i] > sig*thr)
				{
					m_alpha[i] = thr;
					m_alpha[j] = sig*diff - sig*thr;
				}

				if (m_alpha[j] > C)
				{
					m_alpha[j] = C;
					m_alpha[i] = ((sig < 0) ? 0 : diff) - sig*C;
				}

			}
			else
			{
				if (m_alpha[j] < 0)
				{
					m_alpha[i] = diff;
					m_alpha[j] = 0;
				}

				if (sig*m_alpha[i] < sig*(C - thr))
				{
					m_alpha[i] = C - thr;
					m_alpha[j] = (sig < 0) ? C : diff;
				}

			}




			// update G



			double delta_alpha_i = m_alpha[i] - old_alpha_i;
			double delta_alpha_j = m_alpha[j] - old_alpha_j;

			for (int k = 0; k< COLS; k++)
			{
				G[k] += double(Y[i]) * double(Y[k]) * K[i][k] * delta_alpha_i + double(Y[j]) * double(Y[k]) * K[j][k] * delta_alpha_j;

			}

			// update alpha_status and G_bar

			double old_alpha[2] = { old_alpha_i, old_alpha_j };
			int I[2] = { i,j };

			for (int z = 0; z < 2; z++) {
				double sig = (old_alpha[z] == C) ? -1 : 1;

				if ((old_alpha[z] == C) != (m_alpha[I[z]] == C))
				{
					for (int k = 0; k < COLS; k++)
						G_bar[k] += sig*C*double(Y[I[z]]) * double(Y[k]) * K[I[z]][k];

				}
			}

		}

		// calculate rho

		si.rho = calculate_rho(Y, C);
		m_rho = si.rho;

		for (int i = 0; i < m_alpha.size(); i++) {
			if (m_alpha[i] != 0) {
				if (Y[i] < 0)
					si.nNegAlpha++;
				else
					si.nPosAlpha++;
			}
		}

		// calculate objective value
		{
			double v = 0;
			int i;
			for (i = 0; i< COLS; i++)
				v += m_alpha[i] * (G[i] + -1);

			si.obj = v / 2;
		}

		return si;


	}

	void Save(const char *szModelName) {
		KernelParameters param = K.getParam();
		param.m_rho = m_rho;
		K.saveModel(szModelName, m_alpha, param);
	}
};
