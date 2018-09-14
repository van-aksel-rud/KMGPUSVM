#include <windows.h>

// System includes
#include <stdio.h>
#include <assert.h>

#include <cstdio>
#include <iostream>
#include <locale>
#include <iomanip>
#include <chrono>


#include <vector>
#include <string>
#include <list>
#include <set>
#include <algorithm>
#include <numeric>
#include <experimental/filesystem>


#include "CmdLineParser.h"
#include "OnlineKernel.h"
#include "SMO.h"
#include "ConfusionMatrix.h"
#include <set>

double multiClassPrediction(const std::vector<std::string> strModelname, const std::string& testFiles, int nBlockStart, int numBlocks, int mustPredictOne) {


	std::vector<int> classes;
	std::set<int> cls;
	std::vector<OnlineKernel*> vecModels;
	for (int i = 0; i < strModelname.size(); i++) {
		OnlineKernel *pKernel = new OnlineKernel(strModelname[i].c_str());
		vecModels.push_back(pKernel);
		std::cout << std::endl << "Loading model " << strModelname[i];
		pKernel->getParam().print();
		classes.push_back(pKernel->getParam().m_nTarget);
		cls.insert(pKernel->getParam().m_nTarget);
	}

	int nTotalRows = 0;
	int nTestBlocks = numBlocks == 0 ? BatchV::getMaxBatchId(testFiles) : numBlocks;
	std::vector<int> trueLabel;
	for (int i = nBlockStart; i < nBlockStart + nTestBlocks; i++) {
		BatchV test(testFiles, i);
		int nRows = test.getNumRows();
		nTotalRows += nRows;
		for (int j = 0; j < nRows; j++) {
			int l = test.getMultiLabel(j);
			if (cls.find(l) != cls.end())
				trueLabel.push_back(l);
			else
				trueLabel.push_back(-1);
		}
	}

	classes.push_back(-1);

	std::vector<std::vector<double>> predict;
	std::vector<double> predictions;
	predictions.resize(vecModels.size());
	for (int i = 0; i < nTotalRows; i++) {
		predict.push_back(predictions);
	}


	int k = 0;
	int prevScond = 0;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	for (int i = nBlockStart; i < nBlockStart + nTestBlocks; i++) {
		BatchV test(testFiles, i);
		char *buf = new char[test.getTotalSize()];
		SparseVector *pD = (SparseVector*)buf;
		test.read(pD);
		for (int j = 0; j < test.getNumRows(); j++) {
			for (int classId = 0; classId < vecModels.size(); classId++) {
				predict[k][classId] = vecModels[classId]->predict(pD);
			}
			k++;
			pD = (SparseVector*)((char*)pD + pD->m_size);

			std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

			unsigned int currSecond = time_span / 1000;
			if (prevScond != currSecond) {
				ConfusionMatrix conf(predict, k, trueLabel, classes, mustPredictOne);
				conf.print();
				prevScond = currSecond;
			}
		}
		delete[] buf;
	}

	for (int i = 0; i < vecModels.size(); i++)
		delete vecModels[i];


	ConfusionMatrix conf(predict, predict.size(), trueLabel, classes, mustPredictOne);
	return conf.print();
}

int main(int argc, char **argv) {

	{
		std::cout.imbue(std::locale(""));

		std::experimental::filesystem::create_directory("best");


		std::cout << std::endl << "Tuning kernel SVM using SMO" << std::endl;

		strArgument argLearnFiles("--learn-files", "Learn files", "");
		strArgument argTestFiles("--test-files", "Test files", "");
		uintArgument argLearnBlocks("--learn-blocks", "Learn blocks", "0");
		uintArgument argTestBlockStart("--test-block-start", "Test block start", "0");
		uintArgument argTestBlocks("--test-block-num", "Test block number", "0");
		floatArgument argRbfGramma("--rbf-gamma", "RBF kernel gamma", "1");
		floatArgument argKernelC("--kernel-C", "Kernel C", "100");
		doubleArgument argFeatureW("--feature-w", "Weight for each feature", "1");
		uintArgument argRndSeed("--rnd-seed", "Random generator seed", "1");
		uintArgument argKernelCache("--kernel-cache", "Number of rows of K", "100");
		strArgument argModelName("--model-name", "Model file name", "model");
		uintArgument argTargetLabel("--target-label", "Target label", "0");
		uintArgument argPredReqd("--prediction-required", "Must predict one of the classes", "0");

		argument *args[] = { &argLearnFiles, &argTestFiles, &argLearnBlocks, &argTargetLabel,
			&argTestBlockStart , &argTestBlocks, &argRbfGramma, &argKernelC, &argFeatureW, &argRndSeed, &argKernelCache, &argModelName, &argPredReqd
		};

		commandLineParser cmdArgs((argument**)args, sizeof(args) / sizeof(argument*), argv, argv + argc);



		if (argLearnFiles.empty() && argTestFiles.empty()) {
			cmdArgs.printHelp();
			return 0;
		}

		cmdArgs.printArgs();


		if (!argLearnFiles.empty()) {
			double bestError = 1;
			double bestGamma = 0;
			double bestC = 0;

			std::vector<float> vecC = argKernelC;
			std::vector<float> vecG = argRbfGramma;
			std::vector<int> targets = argTargetLabel;
			bool bTesting = targets.size() == 1;

			for (std::vector<float>::const_iterator itGamma = vecG.begin(); itGamma != vecG.end(); itGamma++)
			{
				for (std::vector<float>::const_iterator itC = vecC.begin(); itC != vecC.end(); itC++)
				{

					for (std::vector<int>::iterator itClass = targets.begin(); itClass != targets.end(); itClass++)
					{
						KernelParameters param;
						param.m_nTarget = *itClass;
						param.m_gamma = *itGamma;
						std::vector<double> W = argFeatureW;
						for (int i = 0; (i < W.size() && i < sizeof(param.W) / sizeof(double)); i++)
							param.W[i] = W[i];
						OnlineKernel kernel(param, argLearnFiles, argLearnBlocks, argKernelCache);

						if (!kernel.verify()) {
							std::cout << std::endl << "ERROR: kernel check failed";
							exit(0);
						}

						std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
						{
							std::cout << std::endl << "------------------------------------------------------------------------------------------------------------------";
							std::cout << std::endl << "C=" << *itC << ", gamma=" << *itGamma;
							std::cout << std::endl << "------------------------------------------------------------------------------------------------------------------";
							SMO smo(kernel, kernel.getNumRows());
							SMO::SolutionInfo si = smo.Solve(*itC);
							std::cout << std::endl << "Num Positive Alpha " << si.nPosAlpha;
							std::cout << std::endl << "Num Negative Alpha " << si.nNegAlpha;
							std::cout << std::endl << "Rho " << si.rho;
							smo.Save(argModelName);
						}
						std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
						std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

						std::cout << std::endl << "It took me " << time_span.count() << " seconds." << std::endl;


						//test
						if (!argTestFiles.empty() && bTesting)
						{
							std::vector<int> targets = argTargetLabel;
							std::vector<std::string> testModels;
							for (std::vector<int>::iterator itClass = targets.begin(); itClass != targets.end(); itClass++) {
								std::ostringstream oss;
								oss << argModelName << "_" << *itClass;
								testModels.push_back(oss.str());
							}

							double err = multiClassPrediction(testModels, argTestFiles, argTestBlockStart, argTestBlocks, argPredReqd);
							if (err < bestError) {
								bestError = err;
								bestGamma = *itGamma;
								bestC = *itC;
								for (size_t m = 0; m < testModels.size(); m++) {
									std::ostringstream oss;
									oss << testModels[m] << "_0.dat";
									std::experimental::filesystem::copy(oss.str(), std::string("best/") + oss.str(), std::experimental::filesystem::copy_options::overwrite_existing);
								}
							}
						}

					}
					bTesting = true;
				}
			}
			std::cout << std::endl << "Best error so far " << bestError;
			std::cout << std::endl << "Gamma " << bestGamma;
			std::cout << std::endl << "C     " << bestC;
			std::cout << std::endl;
		}

		if (!argTestFiles.empty() && argLearnFiles.empty())
		{
			std::vector<int> targets = argTargetLabel;
			std::vector<std::string> testModels;
			for (std::vector<int>::iterator itClass = targets.begin(); itClass != targets.end(); itClass++) {
				std::ostringstream oss;
				oss << argModelName << "_" << *itClass;
				testModels.push_back(oss.str());
			}
			double err = multiClassPrediction(testModels, argTestFiles, argTestBlockStart, argTestBlocks, argPredReqd);
		}


	}
	//_CrtDumpMemoryLeaks();


	exit(0);

}





