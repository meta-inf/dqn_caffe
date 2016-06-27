//
// Created by dc on 6/17/15.
//

#include "ReplayMemory.h"
#include <caffe/util/math_functions.hpp>
#include <algorithm>

void ReplayMemory::Add (CPState s0, int action, float reward, CPState s1, bool terminated)
{
	buffer[bufferSize % bufferCapacity] = RMNode{s0, s1, action, reward, terminated};
	vis[bufferSize % bufferCapacity] = 0;
	++bufferSize;
}

void ReplayMemory::Sample (int cnt, std::vector<RMNode> &res)
{
	assert(bufferSize >= (size_t)cnt);

	static int timeStamp = 0;
	++timeStamp;

	res.resize((size_t)cnt);
	for (int i = 0; i < cnt; ++i)
	{
		int id;
		do
		{
			id = rand() % min(bufferSize, bufferCapacity);
			if ((id < bufferSize && id + 4 >= bufferSize) ||
				(id > bufferSize && (id + 4) % bufferCapacity >= bufferSize))
			{
				continue;
			}
			if (vis[id] && vis[id] + 4 >= timeStamp)
			{
				continue;
			}
		} while (false);
		vis[id] = timeStamp;
		res[i] = buffer[id];
	}
}

void ReplayMemory::Init (int bufferSize)
{
	buffer.clear();
	vis.clear();
	buffer.resize((size_t)bufferSize + 1);
	vis.resize((size_t)bufferSize + 1);
	fill(vis.begin(), vis.end(), 0);
	bufferCapacity = bufferSize;
	this->bufferSize = 0;
}
