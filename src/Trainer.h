//
// Created by dc on 6/16/15.
//

#ifndef ATARIDL_LEARNER_H
#define ATARIDL_LEARNER_H

#include <ale_interface.hpp>
#include <caffe/solver.hpp>
#include <caffe/data_layers.hpp>
#include <caffe/util/io.hpp>
#include <vector>
#include <string>

#include "ReplayMemory.h"
#include "GameConsole.h"
#include "TrainerConfig.pb.h"

extern TrainerConfig cfg;

class Trainer
{
public:
	typedef boost::shared_ptr<caffe::Net<float>> pNet;
	typedef boost::shared_ptr<caffe::MemoryDataLayer<float>> pMemoryDataLayer;
	void Init ();
	void Run ();
	void Restore (const std::string &netFile);
	Trainer () {}
protected:
	void Snapshot ();

	// Perform a step of training.
	void UpdateQValues ();

	void EpsSelectAction (CPState state,
						  int &action,
						  float &estimateReward,
						  float eps,
						  bool debug = false);
	void EstimateRewardMinibatch (int id,
								  const std::vector<CPState> &states,
								  std::vector<float> &reward,
								  std::vector<int> &action);

	float GetCurEps ()
	{
		int range = cfg.eps_linear_range();
		if (frameCnt <= range)
		{
			return float(0.1 + 0.9 * (range - frameCnt) / float(range));
		}
		return 0.1f;
	}
	float GetCurLR () { return 0.0003; }

	// `RMSPropStep`
	void SGDStep ();

	void Eval ();
	void Report ();

	static const int CUR = 0, REF = 1;

	// Number of frames simulated
	int frameCnt, episodeCnt;
	float avgQ, avgRewardPerEpisode, rewardLastEpisode;

	pNet QNet[2];
	// Pointer to input layers of each network, so as to feed data into the network
	pMemoryDataLayer LInput[2], LSelector[2], LExpected[2];

	// Pointer to CPU data of respective MemoryDataLayer;
	// size / addr will not change after Init ()
	vector<float> BInput[2], BSelector[2], BExpected[2];
	vector<float> BInputD[2], BSelectorD[2], BExpectedD[2];

	// used by RMSProp
	vector<boost::shared_ptr<caffe::Blob<float>>> rms_tmp, rms_grad, rms_sqr_grad;

	ReplayMemory replayMemory;
	GameConsole gameConsole;

	int availActionCnt;
};

#endif //ATARIDL_LEARNER_H
