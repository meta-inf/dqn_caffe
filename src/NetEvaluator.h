//
// Created by dc on 7/4/15.
//

#ifndef ATARIDL_NETDISPLAYER_H
#define ATARIDL_NETDISPLAYER_H

#include "Trainer.h"

class NetEvaluator : public Trainer
{
public:
	NetEvaluator (const std::string &netParamFile)
	{
		Init();
		Restore(netParamFile);
	}
	void Show ()
	{
		Trainer::Eval();
	}
};


#endif //ATARIDL_NETDISPLAYER_H
