#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <set>

class ConfusionMatrix {
	std::vector<int> m_classes;
	std::map<int, std::map<int, int>> m_mat;
	int m_mustPredictOne;
public:
	ConfusionMatrix(std::vector<std::vector<double>>& pred, int N, std::vector<int>& truth, const std::vector<int>& classes, int mustPredictOne) :
		m_classes(classes),
		m_mustPredictOne(mustPredictOne)
	{
		std::set<int> cls;
		for (std::vector<int>::const_iterator it = classes.begin(); it != classes.end(); it++)
			cls.insert(*it);

		for (size_t i = 0; i < N; i++) {
			int maxI = -1;
			int maxV = (mustPredictOne == 0 ? 0 : -INF);
			for (int j = 0; j < pred[0].size(); j++) {
				if (pred[i][j] > maxV) {
					maxI = j;
					maxV = pred[i][j];
				}
			}

			int t = truth[i];
			if (cls.find(t) == cls.end())
				t = -1;

			m_mat[maxI == -1 ? -1 : classes[maxI]][t]++;
		}
	}
	~ConfusionMatrix() {
	}

	double print() {
		std::cout << std::endl << "Confusion Matrix" << std::endl;

		std::vector<int>::const_iterator ite = m_mustPredictOne ? m_classes.end() - 1 : m_classes.end();

		int total = 0;
		std::cout << std::setw(6) << "";
		for (std::vector<int>::iterator it_pred = m_classes.begin(); it_pred != ite; it_pred++)
			std::cout << std::setw(6) << *it_pred;
		std::cout << std::endl;
		std::cout << std::setw(6) << "";
		for (std::vector<int>::iterator it_pred = m_classes.begin(); it_pred != ite; it_pred++)
			std::cout << "------";
		std::cout << std::endl;

		for (std::vector<int>::iterator it_pred = m_classes.begin(); it_pred != ite; it_pred++) {
			std::cout << std::setw(5) << *it_pred << "|";
			for (std::vector<int>::iterator it_true = m_classes.begin(); it_true != ite; it_true++) {
				int n = m_mat[*it_pred][*it_true];
				std::cout << std::setw(6) << n;
				total += n;
			}
			std::cout << std::endl;
		}

		int tp = 0;
		for (std::vector<int>::const_iterator it = m_classes.begin(); it != m_classes.end(); it++)
			tp += m_mat[*it][*it];

		std::cout << std::endl << "Accuracy " << double(tp) / double(total);
		std::cout << std::endl << "Error " << 1-double(tp) / double(total);

		return 1 - double(tp) / double(total);

		/*
		float precision = float(m_mat[0][0]) / (m_mat[0][0] + m_mat[0][1]);
		float recall = float(m_mat[0][0]) / (m_mat[0][0] + m_mat[1][0]);
		float f1 = 2 * precision*recall / (precision + recall);

		std::cout << "F1 = " << f1 << std::endl;
		std::cout << "Error = " << 1. - double(m_mat[0][0]) / double(m_mat[0][0] + m_mat[0][1]);
		return f1;
		*/
	}
};