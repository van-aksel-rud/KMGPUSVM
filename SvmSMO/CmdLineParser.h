#pragma once
#include <string>
#include <vector>
#include <map>

class commandLineParser;

class argument {
protected:
	friend commandLineParser;
	std::string option;
	std::string desc;
	std::string default;
	std::vector<std::string> m_strVal;
public:
	argument(const char* o, const char *d, const char *def) :
		option(o),
		desc(d),
		default(def)
	{
		m_strVal.push_back("");
	}
	bool empty() const {
		return m_strVal[0].empty();
	}
	bool isDefault() const {
		return default == m_strVal[0];
	}
};

class floatArgument : public argument {
public:
	floatArgument(const char* o, const char *d, const char *def) :
		argument(o,d,def)
	{}
	operator float() const {
		return (float)atof(m_strVal[0].c_str());
	}
	operator std::vector<float>() const {
		std::vector<float> ret;
		for (size_t i = 0; i < m_strVal.size(); i++)
			ret.push_back(atof(m_strVal[i].c_str()));
		return ret;
	}
};

class doubleArgument : public argument {
public:
	doubleArgument(const char* o, const char *d, const char *def) :
		argument(o, d, def)
	{}
	operator double() const {
		return atof(m_strVal[0].c_str());
	}
	operator std::vector<double>() const {
		std::vector<double> ret;
		for (size_t i = 0; i < m_strVal.size(); i++)
			ret.push_back(atof(m_strVal[i].c_str()));
		return ret;
	}
};

class uintArgument : public argument {
public:
	uintArgument(const char* o, const char *d, const char *def) :
		argument(o, d, def)
	{}
	operator size_t() const {
		return atoi(m_strVal[0].c_str());
	}
	operator std::vector<int> () const {
		std::vector<int> ret;
		for (size_t i = 0; i < m_strVal.size(); i++)
			ret.push_back(atoi(m_strVal[i].c_str()));
		return ret;
	}
};

class strArgument : public argument {
public:
	strArgument(const char* o, const char *d, const char *def) :
		argument(o, d, def)
	{}
	operator const char*() const {
		return m_strVal[0].c_str();
	}
	operator const std::string() const {
		return m_strVal[0];
	}
	operator const std::vector<std::string>() const {
		return m_strVal;
	}
};

class commandLineParser {
	argument **m_rargs;
	size_t m_ncnt;

public:
	commandLineParser(argument** args, size_t cnt, char **begin, char **end);
	void printHelp() const;
	void printArgs() const;
};

