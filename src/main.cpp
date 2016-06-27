#include <glog/logging.h>
#include <gflags/gflags.h>
#include "Trainer.h"
#include "NetEvaluator.h"
using namespace std;

DEFINE_string(restore_from, "__none__", "network snapshot to be restored");
DEFINE_string(config, "../src/trainer.prototxt", "trainer configuration");
DEFINE_bool(playback, false, "display trained model");
DEFINE_bool(display_screen, false, "display emulator screen");
DEFINE_int32(gpu_id, -1, "GPU Device ID to use. -1 for CPU");

TrainerConfig cfg;

int main_playback ()
{
	NetEvaluator eva(FLAGS_restore_from);
	eva.Show();
	return 0;
}

int main (int argc, char *argv[])
{
//	testReplay();
	google::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	if (FLAGS_gpu_id == -1)
	{
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
	}
	else
	{
		LOG(INFO) << "Use GPU " << FLAGS_gpu_id << endl;
		caffe::Caffe::SetDevice(FLAGS_gpu_id);
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
	}

	caffe::Caffe::Get().set_random_seed(1000000007);

#define OUTPUT(X) #X << " = " << X << " "

	caffe::ReadProtoFromTextFileOrDie(FLAGS_config, &cfg);
	
	if (FLAGS_playback)
	{
		FLAGS_display_screen = !FLAGS_display_screen;
		return main_playback();
	}

	Trainer learner;
	learner.Init();
	if (FLAGS_restore_from != "__none__")
	{
		learner.Restore(FLAGS_restore_from);
	}

	learner.Run();

	return 0;
}
