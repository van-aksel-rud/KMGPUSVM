#include "CmdLineParser.h"
#include <iostream>
#include <locale>
#include <iomanip>


std::vector<std::string> getCmdOptionStr(char ** begin, char ** end, const std::string & option, const std::string& default)
{
	std::vector<std::string> ret;

	char ** itr = begin;
	do {
		itr = std::find(itr, end, option);
		if (itr != end && ++itr != end)
		{
			ret.push_back(std::string(*itr));
			++itr;
		}
		else
			break;
	} while (true);

	if (ret.empty())
		ret.push_back(default);
	return ret;
}

commandLineParser::commandLineParser(argument** args, size_t c, char **begin, char **end) :
	m_rargs(args),
	m_ncnt(c)
{
	for (size_t i = 0; i < m_ncnt; i++) {
		m_rargs[i]->m_strVal = getCmdOptionStr(begin, end, m_rargs[i]->option, m_rargs[i]->default);
	}
}

void commandLineParser::printHelp() const {
	for (size_t i = 0; i < m_ncnt; i++) {
		std::cout << std::endl << std::left << std::setw(50) << m_rargs[i]->desc << std::setw(30) << m_rargs[i]->option << " (" << m_rargs[i]->default << ")";
	}
}

void commandLineParser::printArgs() const {
	for (size_t i = 0; i < m_ncnt; i++) {
		std::cout << std::endl << std::left << std::setw(50) << m_rargs[i]->desc << std::setw(30) << m_rargs[i]->option;
		for (int j = 0; j < m_rargs[i]->m_strVal.size(); j++)
			std::cout << m_rargs[i]->m_strVal[j] << " ";
	}
}
