//
// Created by dc on 6/17/15.
//

#ifndef ATARIDL_GAMECONSOLE_H
#define ATARIDL_GAMECONSOLE_H

#include <string>
#include <ale_interface.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>

const int screenWidth = 84, screenHeight = 84;

struct Frame
{
	static const int flen = screenWidth * screenHeight;
	float data[flen];
	Frame (bool init = true)
	{
		if (init)
		{
			std::fill(data, data + flen, 0);
		}
	}
};

typedef boost::shared_ptr<const Frame> CPFrame;

struct State
{
	static const int nframe = 4;
	static const int size = nframe * Frame::flen;
	CPFrame f[4];

	State &operator= (const State &b)
	{
		for (int i = 0; i < nframe; ++i) f[i] = b.f[i];
		return *this;
	}
	template <typename FIter>
	void LoadTo (FIter dst) const
	{
		for (int d = 0; d < nframe; ++d)
		{
			std::copy(f[d]->data, f[d]->data + Frame::flen, dst + d * Frame::flen);
		}
	}
};

typedef boost::shared_ptr<const State> CPState;

class GameConsole
{
public:
	void Init (const std::string &romName, int &availableMoves);
	void Reset ();
	void Act (int action, float &reward, CPState &nstate, bool &terminated);
	bool Terminated ();
	std::string GetName (int action) { return action_to_string(actions[action]); }
	CPState CurState ();
	GameConsole ();

protected:
	void SaveState ();
	void UpdateScreen ();

	CPFrame buffer[4];
	ActionVect actions;
	ALEInterface ale;
};

#endif //ATARIDL_GAMECONSOLE_H
