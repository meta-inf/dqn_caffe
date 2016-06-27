//
// Created by dc on 6/16/15.
//

#include "Trainer.h"
#include <caffe/proto/caffe.pb.h>
#include <glog/logging.h>

using namespace std;
using namespace caffe;

template <typename T> inline T sqr (T x) { return x * x; }

void Trainer::Init ()
{
	replayMemory.Init(cfg.replay_memory_capacity());
	gameConsole.Init(cfg.rom_name(), availActionCnt);

	NetParameter netParameter;
	ReadProtoFromTextFileOrDie("../src/dqn.prototxt", &netParameter);
	QNet[CUR] = pNet(new Net<float>(netParameter));
	QNet[REF] = pNet(new Net<float>(netParameter));
	LOG(INFO) << "Network created." << endl;

	for (int i = 0; i < 2; ++i)
	{
		LInput[i] = boost::dynamic_pointer_cast<MemoryDataLayer<float>>(QNet[i]->layer_by_name("InputDataLayer"));
		LSelector[i] = boost::dynamic_pointer_cast<MemoryDataLayer<float>>(QNet[i]->layer_by_name("SelectorInput"));
		LExpected[i] = boost::dynamic_pointer_cast<MemoryDataLayer<float>>(QNet[i]->layer_by_name("Expected"));
		assert(LInput[i] && LSelector[i] && LExpected[i]);
#define blobSize(X) (size_t)(X->batch_size() * X->height() * X->width() * X->channels())
#define SETUPBuffer(X) B ## X [i].resize(blobSize(L ## X[i])); B ## X ## D[i].resize(blobSize(L ## X[i]));
		SETUPBuffer(Input);
		SETUPBuffer(Selector);
		SETUPBuffer(Expected);
#undef blobSize
#undef SETUPBuffer
		LSelector[i]->Reset(BSelector[i].data(), BSelectorD[i].data(), cfg.minibatch_size());
		LExpected[i]->Reset(BExpected[i].data(), BExpectedD[i].data(), cfg.minibatch_size());
	}

	const auto& par = QNet[CUR]->params();
	rms_grad.resize(par.size());
	rms_tmp .resize(par.size());
	rms_sqr_grad.resize(par.size());
	for (size_t i = 0; i < rms_grad.size(); ++i)
	{
		rms_grad[i] = boost::shared_ptr<Blob<float>>(new Blob<float>(par[i]->shape()));
		rms_tmp[i]  = boost::shared_ptr<Blob<float>>(new Blob<float>(par[i]->shape()));
		rms_sqr_grad[i]  = boost::shared_ptr<Blob<float>>(new Blob<float>(par[i]->shape()));
	}

	frameCnt = 0;
	avgQ = avgRewardPerEpisode = rewardLastEpisode = 0;
}


void Trainer::Eval ()
{
#define OUTPUT(X) #X << " = " << X << " "
	gameConsole.Reset();
	CPState state0 = gameConsole.CurState();
	LOG(INFO) << "EVAL FrameCnt = " << frameCnt << endl;
	int cntRound = 0;
	float sumReward = 0, lastEpisodeReward = 0;
	for (int i = 0; i < cfg.eval_time(); ++i)
	{
		LOG(INFO) << "\tIteration " << i;
		int action;
		float reward, reward_e;
		bool terminated;
		EpsSelectAction(state0, action, reward_e, cfg.eval_eps(), i % cfg.dump_action_freq() == 1);
		gameConsole.Act(action, reward, state0, terminated);
		sumReward += reward;
		lastEpisodeReward += reward;
		LOG(INFO) << "\t" << OUTPUT(reward) << OUTPUT(reward_e) << OUTPUT(terminated) << OUTPUT(gameConsole.GetName(action)) << endl;
		if (terminated)
		{
			lastEpisodeReward = 0;
			++cntRound;
			gameConsole.Reset();
			state0 = gameConsole.CurState();
			continue;
		}
	}
	LOG(INFO) << "Total reward = " << sumReward << "(" << lastEpisodeReward << ") /" << cntRound << endl;
	state0.reset();
}

void Trainer::Report ()
{
	float lr = GetCurLR();
	float eps = GetCurEps();
	LOG(INFO) << "REPORT " << OUTPUT(frameCnt) << "\n\t"
	<< OUTPUT(eps) << OUTPUT(avgRewardPerEpisode)
	<< "\n\t" << OUTPUT(avgQ) << OUTPUT(rewardLastEpisode) << OUTPUT(episodeCnt) << endl;
	if ((frameCnt / cfg.log_freq()) % cfg.dump_net_freq() != 1)
		return;

	ostringstream buf;
	for (int i = 0; i < QNet[CUR]->params().size(); ++i)
	{
		int cnt = QNet[CUR]->params()[i]->count();
		const float *dt = QNet[CUR]->params()[i]->cpu_data();
		float avg = 0;
		for (int j = 0; j < cnt; ++j) avg += dt[j];
		buf << ": " << avg / (float)cnt;
	}
	LOG(INFO) << "\tNetParams:\n\t" << buf.str() << endl;
}

void Trainer::Run ()
{
	CPState state0(nullptr), state1(gameConsole.CurState());
	int nextEvalTime = cfg.eval_start_time();
	for (; frameCnt < cfg.total_iterations(); )
	{
		if (frameCnt % cfg.log_freq() == 0)
		{
			this->Report();
		}
		if (frameCnt % cfg.snapshot_freq() == 0 && frameCnt > cfg.snapshot_freq())
		{
			this->Snapshot();
		}
		if (frameCnt == nextEvalTime)
		{
			this->Eval();
			gameConsole.Reset();
			state1 = gameConsole.CurState();
			nextEvalTime += cfg.eval_freq();
			// frameCnt is unchanged after evaluation. 
			continue;
		}

		int action;
		float reward, reward_e;
		bool terminated;
		EpsSelectAction(state1, action, reward_e, GetCurEps());
		state0 = state1;
		gameConsole.Act(action, reward, state1, terminated);
		++frameCnt;
		rewardLastEpisode += reward;
		replayMemory.Add(state0, action, reward, state1, terminated);

		if (terminated)
		{
			++episodeCnt;
			gameConsole.Reset();
			state1 = gameConsole.CurState();
			avgRewardPerEpisode = avgRewardPerEpisode * 0.9f + rewardLastEpisode * 0.1f;
			rewardLastEpisode = 0;
			continue;
		}

		if (frameCnt > cfg.learn_start_time())
		{
			if (frameCnt % cfg.update_freq() == 0)
			{
				UpdateQValues();
			}
			if (frameCnt % cfg.nn_sync_freq() == 0)
			{
				auto &dst = QNet[REF]->params();
				const auto &src = QNet[CUR]->params();
				for (size_t i = 0; i < src.size(); ++i)
				{
					dst[i]->CopyFrom(*src[i], false, false);
				}
			}
		}
	}
}

void Trainer::EpsSelectAction (CPState state, int &action, float &estimateReward, float eps, bool debug)
{
	float p;
	caffe_rng_uniform<float>(1, 0, 1, &p);
	if (p < eps)
	{
		caffe_rng_uniform<float>(1, 0, availActionCnt - 1, &p);
		action = int(p + 1e-5) % availActionCnt;
		estimateReward = 0;
		return;
	}

	state->LoadTo(BInput[CUR].begin());
	LInput[CUR]->Reset(BInput[CUR].data(), BInputD[CUR].data(), cfg.minibatch_size());
	QNet[CUR]->ForwardPrefilled(NULL);
	const boost::shared_ptr<Blob<float>> output = QNet[CUR]->blob_by_name("output");
	assert(output);
	
	if (debug)
	{
		ostringstream buf;
		for (int c = 0; c < availActionCnt; ++c)
			buf << gameConsole.GetName(c) << "\t";
		buf << endl << "\t";
		for (int c = 0; c < availActionCnt; ++c)
			buf << output->data_at(0, c, 0, 0) << "\t";
		buf << endl;
		LOG(INFO) << "\tAction rewards:\n\t" << buf.str(); 
	}

	int bst = 0;
	for (int cur = 1; cur < availActionCnt; ++cur)
	{
		if (output->data_at(0, cur, 0, 0) >
				output->data_at(0, bst, 0, 0))
		{
			bst = cur;
		}
	}
	action = bst;
	estimateReward = output->data_at(0, action, 0, 0);

	avgQ *= 0.9; avgQ += 0.1 * abs(estimateReward);
}

void Trainer::UpdateQValues ()
{

	// Zero out previous gradients.
	for (int i = 0; i < QNet[CUR]->params().size(); ++i)
	{
		boost::shared_ptr<Blob<float>> blob = QNet[CUR]->params()[i];
		switch (Caffe::mode())
		{
			case Caffe::CPU:
				caffe_set(blob->count(), 0.0f, blob->mutable_cpu_diff());
				break;
			case Caffe::GPU:
#ifndef CPU_ONLY
				caffe_gpu_set(blob->count(), 0.0f, blob->mutable_gpu_diff());
#else
				NO_GPU;
#endif
		}
	}

	vector<RMNode> samples;
	vector<float> outputReward;
	vector<int> outputAction;

	replayMemory.Sample(cfg.minibatch_size(), samples);

	// calculate Exp[i] = reward[i] | (Q(s1[i] | ref) * gamma + reward[i])
	vector<CPState> batch;
	for (int s = 0; s < cfg.minibatch_size(); ++s)
	{
		if (!samples[s].terminated)
		{
			batch.push_back(samples[s].s1);
		}
	}
	EstimateRewardMinibatch(REF, batch, outputReward, outputAction);

	fill(BSelector[CUR].begin(), BSelector[CUR].end(), 0.0f);
	fill(BExpected[CUR].begin(), BExpected[CUR].end(), 0.0f);
	for (int s = 0, idx = 0; s < cfg.minibatch_size(); ++s)
	{
		float expected;
		if (samples[s].terminated)
		{
			expected = samples[s].reward;
		}
		else
		{
			float nxtScore = outputReward[idx];
			++idx;
			expected = samples[s].reward + cfg.gamma() * nxtScore;
		}
		BSelector[CUR][s * 18 + samples[s].action] = 1.0f;
		BExpected[CUR][s * 18 + samples[s].action] = expected;
	};

	// feed Exp & Selector into QNet[CUR]
	for (int s = 0; s < cfg.minibatch_size(); ++s)
	{
		samples[s].s0->LoadTo(BInput[CUR].begin() + s * State::size);
	}
	LInput[CUR]->Reset(BInput[CUR].data(), BInputD[CUR].data(), cfg.minibatch_size());
	LSelector[CUR]->Reset(BSelector[CUR].data(), BSelectorD[CUR].data(), cfg.minibatch_size());
	LExpected[CUR]->Reset(BExpected[CUR].data(), BExpectedD[CUR].data(), cfg.minibatch_size());

	QNet[CUR]->ForwardPrefilled(NULL);

	// Scaffolding
	SGDStep();
}

void Trainer::SGDStep ()
{
	pNet net = QNet[CUR];
	net->Backward();
	float lr = GetCurLR();
	const vector<boost::shared_ptr<Blob<float>>>& params = net->params();
	for (size_t i = 0; i < params.size(); ++i)
	{
		int n = params[i]->count();
		switch (Caffe::mode())
		{
			case Caffe::CPU:
			{
				caffe_cpu_axpby<float>(n, 0.05, params[i]->cpu_diff(), 0.95, rms_grad[i]->mutable_cpu_data());

				caffe_sqr<float>(n, params[i]->cpu_diff(), rms_tmp[i]->mutable_cpu_data());
				caffe_cpu_axpby<float>(n, 0.05, rms_tmp[i]->cpu_data(), 0.95, rms_sqr_grad[i]->mutable_cpu_data());

				caffe_sqr<float>(n, rms_grad[i]->cpu_data(), rms_tmp[i]->mutable_cpu_data());
				caffe_cpu_axpby<float>(n, 1, rms_sqr_grad[i]->cpu_data(), -1, rms_tmp[i]->mutable_cpu_data());
				// rms_tmp[data] = Var[params_diff]

				caffe_add_scalar<float>(n, 0.01, rms_tmp[i]->mutable_cpu_data());
				caffe_powx<float>(n, rms_tmp[i]->cpu_data(), 0.5, rms_tmp[i]->mutable_cpu_data());
				caffe_div<float>(n, params[i]->cpu_diff(), rms_tmp[i]->cpu_data(), rms_tmp[i]->mutable_cpu_diff());
				caffe_cpu_scale<float>(n, lr * cfg.update_freq(), rms_tmp[i]->cpu_diff(), params[i]->mutable_cpu_diff());
				break;
			}
			case Caffe::GPU:
			{
#ifndef CPU_ONLY
				caffe_gpu_axpby<float>(n, 0.05, params[i]->gpu_diff(), 0.95, rms_grad[i]->mutable_gpu_data());

				caffe_gpu_powx<float>(n, params[i]->gpu_diff(), 2, rms_tmp[i]->mutable_gpu_data());
				caffe_gpu_axpby<float>(n, 0.05, rms_tmp[i]->gpu_data(), 0.95, rms_sqr_grad[i]->mutable_gpu_data());

				caffe_gpu_powx<float>(n, rms_grad[i]->gpu_data(), 2, rms_tmp[i]->mutable_gpu_data());
				caffe_gpu_axpby<float>(n, 1, rms_sqr_grad[i]->gpu_data(), -1, rms_tmp[i]->mutable_gpu_data());
				// rms_tmp[data] = Var[params_diff]

				caffe_gpu_add_scalar<float>(n, 0.01, rms_tmp[i]->mutable_gpu_data());
				caffe_gpu_powx<float>(n, rms_tmp[i]->gpu_data(), 0.5, rms_tmp[i]->mutable_gpu_data());
				caffe_gpu_div<float>(n, params[i]->gpu_diff(), rms_tmp[i]->gpu_data(), rms_tmp[i]->mutable_gpu_diff());
				caffe_gpu_scale<float>(n, lr * cfg.update_freq(), rms_tmp[i]->gpu_diff(), params[i]->mutable_gpu_diff());
#endif
			}
		}
	}
	net->Update();
}

void Trainer::EstimateRewardMinibatch (int id, const vector<CPState> &states, vector<float> &reward, vector<int> &action)
{
	assert(states.size() <= cfg.minibatch_size());
	for (size_t i = 0; i < states.size(); ++i)
	{
		states[i]->LoadTo(BInput[id].begin() + i * State::size);
	}
	LInput[id]->Reset(BInput[id].data(), BInputD[id].data(), cfg.minibatch_size());
	QNet[id]->ForwardPrefilled(NULL);
	const boost::shared_ptr<Blob<float>> output = QNet[id]->blob_by_name("output");
	assert(output);
	reward.resize(states.size());
	action.resize(states.size());
	for (size_t i = 0; i < states.size(); ++i)
	{
		reward[i] = -1e10f;
		for (int j = 0; j < availActionCnt; ++j)
		{
			float cur = output->data_at(i, j, 0, 0);
			if (cur > reward[i])
			{
				reward[i] = cur;
				action[i] = j;
			}
		}
	}
}

void Trainer::Snapshot ()
{
	NetParameter *netParameter = new NetParameter;
	QNet[CUR]->ToProto(netParameter);
	ostringstream sout;
	sout << "param_at_" << frameCnt << ".bin";
	LOG(INFO) << "Saving network parameters to " << sout.str() << endl;
	WriteProtoToBinaryFile(*netParameter, sout.str());
	delete netParameter;
}

void Trainer::Restore (const string &netFile)
{
	NetParameter *netParameter = new NetParameter;
	ReadProtoFromBinaryFileOrDie(netFile, netParameter);
	QNet[CUR]->CopyTrainedLayersFrom(*netParameter);
	QNet[REF]->CopyTrainedLayersFrom(*netParameter);
}
