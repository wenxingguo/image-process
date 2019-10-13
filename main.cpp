#include "img.h"

double f(uchar x, uchar n = 50) {
	if (x >= n) return (x - n) * (x - n) * (255.0 / (255.0 - n) / (255.0 - n));
	return 0;
}

int main(int argc, char** argv) {
	if (!argv[1]) {
		fl_alert("please intput a file name!!!");
		return -1;
	}

	CV_IMAGE img(argv[1], 0);

	if (!img.data) {
		fl_alert("such file is not exist!!");
		return -1;
	}

	CV_IMAGE outimg(img);

	process::fft_gray_img(img, outimg);
	auto func = [](double x) -> double { return f(x, 100); };

	process::transfer_gray_value(outimg, func);
	outimg.add_show_list();
	img.add_show_list();
	//outimg.imgsave("fft_tem.jpeg");
	CV_IMAGE::imshow();
}
