//
// Created by dc on 6/17/15.
//

#include "GameConsole.h"
#include "TrainerConfig.pb.h"
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

DECLARE_bool(display_screen);
extern TrainerConfig cfg;

GameConsole::GameConsole ()
{
	for (int i = 0; i < 4; ++i)
	{
		buffer[i] = CPFrame(new Frame());
	}
}

void GameConsole::Init (const std::string &romName, int &availableMoves)
{
	ale.loadROM(romName);
	actions = ale.getLegalActionSet();
	availableMoves = (int)actions.size();
	ale.reset_game();
	if (FLAGS_display_screen)
	{
		cv::namedWindow("screen");
	}
}

void GameConsole::Act (int action, float &reward, CPState &nstate, bool &terminated)
{
	reward = 0;
	for (int i = 1; i <= cfg.action_repeat(); ++i)
	{
		reward += ale.act(actions[action]);
		if (FLAGS_display_screen)
		{
			UpdateScreen();
		}
	}
	SaveState();
	nstate = CurState();
	terminated = Terminated();
}

void GameConsole::Reset ()
{
	ale.reset_game();
	for (int i = 0; i < 4; ++i)
	{
		buffer[i] = CPFrame(new Frame());
	}
}

void NTSC2RGB (unsigned char ntsc, unsigned char &r, unsigned char &g, unsigned char &b)
{
	const int palette[] = {
			0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
			0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
			0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
			0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
			0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
			0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
			0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
			0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
			0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
			0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
			0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
			0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
			0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
			0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
			0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
			0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
			0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
			0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
			0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
			0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
			0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
			0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
			0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
			0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
			0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
			0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
			0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
			0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
			0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
			0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
			0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
			0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0
	};
	int v = palette[ntsc];
	r = (unsigned char)((v & 0xFF0000) >> 16);
	g = (unsigned char)((v & 0x00FF00) >> 8);
	b = (unsigned char)(v & 0x0000FF);
}

void GameConsole::UpdateScreen ()
{
	ALEScreen screen = ale.getScreen();

	// NTSC -> rgb
	cv::Mat image((int)screen.height(), (int)screen.width(), CV_8UC3);
	for (int i = 0; i < screen.height(); ++i)
		for (int j = 0; j < screen.width(); ++j)
		{
			auto &vec = image.at<cv::Vec3b>(i, j);
			NTSC2RGB(screen.get(i, j), vec[0], vec[1], vec[2]);
		}

	cv::Mat tmp2 = image;
	cv::resize(tmp2, tmp2, cv::Size_<int>(490, 490));
	cv::imshow("screen", tmp2);
	cv::waitKey(cfg.playback_display_rate());
}

void GameConsole::SaveState ()
{
	ALEScreen screen = ale.getScreen();

	// NTSC -> rgb
	cv::Mat image((int)screen.height(), (int)screen.width(), CV_8UC3);
	for (int i = 0; i < screen.height(); ++i)
		for (int j = 0; j < screen.width(); ++j)
		{
			auto &vec = image.at<cv::Vec3b>(i, j);
			NTSC2RGB(screen.get(i, j), vec[0], vec[1], vec[2]);
		}

	// extract luminance
	cv::Mat tmp;
	cv::resize(image, tmp, cv::Size_<int>(screenHeight, screenWidth));
	cv::cvtColor(tmp, image, CV_RGB2YCrCb);

	for (int i = 3; i >= 1; --i)
		buffer[i] = buffer[i - 1];

	Frame *frame = new Frame(false);
	for (int i = 0; i < screenHeight; ++i)
		for (int j = 0; j < screenWidth; ++j)
		{
			frame->data[i * screenWidth + j] = image.at<cv::Vec3b>(i, j)[0];
		}
	buffer[0] = CPFrame(frame);
}

CPState GameConsole::CurState ()
{
	State *dst = new State{};
	for (int d = 0; d < 4; ++d)
	{
		dst->f[d] = buffer[d];
	}
	return boost::shared_ptr<State>(dst);
}

bool GameConsole::Terminated ()
{
	return ale.game_over();
}