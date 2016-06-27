//
// Created by dc on 6/17/15.
//

#ifndef ATARIDL_REPLAYMEMORY_H
#define ATARIDL_REPLAYMEMORY_H


#include <boost/smart_ptr/shared_ptr.hpp>
#include "GameConsole.h"

struct RMNode
{
	CPState s0, s1;
	int action;
	float reward;
	bool terminated;
};

class ReplayMemory
{
public:
	void Add (CPState s0, int action, float reward, CPState s1, bool terminated);
	void Sample (int cnt, std::vector<RMNode> &res);
	void Init (int bufferSize);
protected:
	std::vector<RMNode> buffer;
	std::vector<int> vis;
	int bufferCapacity, bufferSize;
};


#endif //ATARIDL_REPLAYMEMORY_H
