#include "img.h"
/*
	-------------------------------  X
	|
	|
	|
	|           IMAGE
	|
	|
	|
	|
	|

	Y

 */
CV_IMAGE::CV_IMAGE(cv::Mat& img) :cols(img.cols),
rows(img.rows), channels(img.channels()) {
	if (!img.data) {
		return;
	}
	else {
		int size = rows * cols * channels;
		data = new uchar[size];
		for (int i = 0; i < size; ++i) {
			data[i] = img.data[i];
		}
	}
}
void CV_IMAGE::convert_CV_img(cv::Mat& img) {
	if (!img.data) return;
	int channels = img.channels();
	uchar* p = img.data;
	uchar* q = p + channels * img.cols * img.rows;
	for (; p < q; p += channels) {
		*p = *p ^ *(p + 2);
		*(p + 2) = *p ^ *(p + 2);
		*p = *p ^ *(p + 2);
	}
}

CV_IMAGE::CV_IMAGE(int in_cols, int in_rows, int in_channels) :cols(in_cols),
rows(in_rows), channels(in_channels) {
	data = new uchar[rows * cols * channels]();
}

CV_IMAGE::CV_IMAGE(char* filename, int flag) {
	cv::Mat img = cv::imread(filename, flag);
	convert_CV_img(img);
	if (!img.data) {
		return;
	}
	else {
		cols = img.cols;
		rows = img.rows;
		channels = img.channels();
		int size = cols * rows * channels;
		data = new uchar[size];
		for (int i = 0; i < size; ++i) {
			data[i] = img.data[i];
		}
	}
}

CV_IMAGE::CV_IMAGE() : cols(0), rows(0), channels(0), data(nullptr) {
}

CV_IMAGE::CV_IMAGE(CV_IMAGE& img) : cols(img.cols), rows(img.rows), channels(img.channels) {
	data = new uchar[cols * rows * channels]();
}

CV_IMAGE::CV_IMAGE(CV_IMAGE& img, int pose_x, int pose_y, int domain_x, int domain_y) : cols(domain_x), rows(domain_y), channels(img.channels)
{
	data = new uchar[channels * rows * cols];
	int i = 0;
	for (int y = pose_y; y < pose_y + domain_y; ++y) {
		for (int x = pose_x; x < pose_x + domain_x; ++x) {
			for (int j = 0; j < img.channels; ++j) {
				data[i * channels + j] = img.data[((y - 1) * img.cols + x - 1) * channels + j];
			}
			i++;
		}
	}
}
CV_IMAGE::CV_IMAGE(CONV_CORE& core) :CV_IMAGE(core.cols, core.rows, core.channels) {

}

CV_IMAGE::~CV_IMAGE() {
	if (data) {
		delete data;
		data = nullptr;
	}
}

void CV_IMAGE::add_show_list() {
	img_show.push_back(this);

}

int CV_IMAGE::imshow() {
	if (img_show.size() == 0) {
		fl_alert("No image to show");
		return 0;
	}
	Fl_Window** windows_list = new Fl_Window * [img_show.size()];
	for (int i = 0; i < img_show.size(); ++i) {
		windows_list[i] = new Fl_Window(img_show[i]->cols, img_show[i]->rows);
		Fl_Box* b = new Fl_Box(0, 0, img_show[i]->cols, img_show[i]->rows);
		Fl_RGB_Image* box_img = new Fl_RGB_Image(img_show[i]->data, img_show[i]->cols, img_show[i]->rows, img_show[i]->channels);
		b->image(box_img);
		windows_list[i]->end();
		windows_list[i]->show();
	}
	delete[] windows_list;
	return Fl::run();
}

void CV_IMAGE::imgsave(const std::string s) {
	cv::imwrite(s, cv::Mat(cols, rows, 0, data));
}

std::vector<CV_IMAGE*> CV_IMAGE::img_show{};


CONV_CORE::CONV_CORE(int size, int in_channels) :cols(size), rows(size), channels(in_channels) {
	data = new double[size * size * in_channels]();
}

CONV_CORE::CONV_CORE(double* in_data, int size, int in_channels) :
	data(in_data), channels(in_channels), rows(size), cols(size) {

}

CONV_CORE::CONV_CORE(CONV_CORE& r_core, CONV_CORE& g_core, CONV_CORE& b_core) : channels(3), rows(r_core.rows),
cols(r_core.cols) {
	int size = cols * rows;
	data = new double[size * channels];
	for (int i = 0; i < size; ++i) {
		data[i * channels + 0] = r_core.data[i];
		data[i * channels + 1] = g_core.data[i];
		data[i * channels + 2] = b_core.data[i];
	}
}

CONV_CORE::CONV_CORE(CONV_CORE& r_core, CONV_CORE& g_core, CONV_CORE& b_core, double in_scal) :channels(3), rows(r_core.rows),
cols(r_core.cols) {
	int size = cols * rows;
	data = new double[size * channels];
	for (int i = 0; i < size; ++i) {
		data[i * channels + 0] = r_core.data[i] * in_scal;
		data[i * channels + 1] = g_core.data[i] * in_scal;
		data[i * channels + 2] = b_core.data[i] * in_scal;
	}
}

CONV_CORE::~CONV_CORE() {
	if (data) {
		delete data;
		data = nullptr;
	}
}


void process::reverse(CV_IMAGE& img) {
	int channels = img.channels;
	uchar back = 255;
	uchar* p = img.data;
	uchar* q = p + channels * img.rows * img.cols;
	for (; p < q; ++p) {
		*p = back - *p;
	}
}

void process::log_convert(CV_IMAGE& img, int c) {
	int channels = img.channels;
	uchar* p = img.data;
	uchar* q = p + channels * img.cols * img.rows;
	for (; p < q; ++p) {
		*p = c * std::log10(*p + 1);
	}
}

void process::gamma_convert(CV_IMAGE& img, double c, double gamma) {
	uchar* p = img.data;
	uchar* q = p + img.channels * img.cols * img.rows;
	for (; p < q; ++p) {
		*p = c * std::pow(*p, gamma);
	}
}

void process::get_color_channel(CV_IMAGE& img, CV_IMAGE& outimg, CV_IMAGE::color_channels color)
{
	if (outimg.channels != 1) std::cerr << "gray image's channels is not 1 !!!" << std::endl;
	int channels = img.channels;
	int size = outimg.cols * outimg.rows;
	outimg.data = new uchar[size];
	for (int i = 0; i < size; ++i) {
		*(outimg.data + i) = *(img.data + channels * i + color);
	}
}

void process::rgb2image(CV_IMAGE& rimg, CV_IMAGE& gimg, CV_IMAGE& bimg, CV_IMAGE& outimg) {
	int size = outimg.cols * outimg.rows;
	outimg.data = new uchar[size * outimg.channels];

	for (int i = 0; i < size; ++i) {
		outimg.data[outimg.channels * i + 0] = rimg.data[i];
		outimg.data[outimg.channels * i + 1] = gimg.data[i];
		outimg.data[outimg.channels * i + 2] = bimg.data[i];
	}

}

void process::rgb2gray(CV_IMAGE& img, CV_IMAGE& outimg) {
	if (outimg.channels != 1) std::cerr << "gray image's channels is not 1 !!!" << std::endl;
	int size = img.cols * img.rows;
	outimg.data = new uchar[size];
	for (int i = 0; i < size; ++i) {
		outimg.data[i] = (img.data[i * img.channels] + img.data[i * img.channels + 1] + img.data[i * img.channels + 2]) / 3.0;
	}
}

int* process::hist(CV_IMAGE& img) {
	//r > g > b
	int* array = new int[256 * img.channels](); //切记初始化
	//
	//
	int size = img.cols * img.rows;
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < img.channels; ++j) {
			array[img.data[i * img.channels + j] + j * 256]++;
		}
	}
	for (int i = 1; i < 256; ++i) {
		//累加起来，当前位置值是前面全部值的和
		for (int j = 0; j < img.channels; ++j) {
			array[i + j * 256] += array[i - 1 + j * 256];
		}
	}
	return array;
}

void process::hist_equa(CV_IMAGE& img, CV_IMAGE& outimg) {
	int size = img.cols * img.rows;
	int* hist_array = process::hist(img); //记得释放内存
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < img.channels; ++j) {
			outimg.data[i * outimg.channels + j] = 255.0 / size * hist_array[img.data[i * img.channels + j] + j * 256];
		}
	}
	delete hist_array;
}

void process::hist_spec(CV_IMAGE& img, CV_IMAGE& outimg, int* spec_hist_array) {
	int* hist_array = hist(img);
	int size = img.rows * img.cols;
	uchar temp_gsv = 0;
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < img.channels; ++j) {
			temp_gsv = img.data[i * img.channels + j];
			for (uchar k = 255; k >= 0; --k) {
				if (spec_hist_array[k + j * 256] <= hist_array[temp_gsv + j * 256]) {
					outimg.data[i * outimg.channels + j] = k;
					break;
				}
			}
		}
	}
	delete hist_array;
}

void gen_spec_array_density(CV_IMAGE& img, int* hist_array, double (*f)(int)) {
	int size = img.rows * img.cols;
	double temp_sum = 0;
	for (int i = 0; i < 256; ++i) {
		hist_array[i] = temp_sum;
		temp_sum += f(i);
	}
	switch (img.channels)
	{
	case 1:
		for (int i = 0; i < 256; ++i) {
			hist_array[i] = double(hist_array[i]) / hist_array[255] * size;
		}
		break;

	case 3:
		for (int i = 0; i < 256; ++i) {
			hist_array[i] = double(hist_array[i]) / hist_array[255] * size;
			for (int j = 1; j < img.channels; ++j) {
				hist_array[i + j * 256] = hist_array[i];
			}
		}
		break;
	}
}

void gen_spec_array_disturb(CV_IMAGE& img, int* hist_array, double (*f)(int)) {
	int size = img.cols * img.rows;
	double temp = f(255);
	switch (img.channels)
	{
	case 1:
		for (int i = 0; i < 256; ++i) {
			hist_array[i] = f(i) / temp * size;
		}
		break;
	case 3:
		for (int i = 0; i < 256; ++i) {
			hist_array[i] = f(i) / temp * size;
			for (int j = 1; j < img.channels; ++j) {
				hist_array[i + j * 256] = hist_array[i];
			}
		}
		break;
	}
}

int* process::gen_spec_hist(CV_IMAGE& img, double (*f)(int), process::prob_func_type f_type) {
	int* spec_array = new int[img.channels * 256]();
	switch (f_type)
	{
	case process::PROB_DENSITY:
		gen_spec_array_density(img, spec_array, f);
		break;

	case process::PROB_DISTRIB:
		gen_spec_array_disturb(img, spec_array, f);
		break;
	}
	return spec_array;
}

void process::get_domian(CV_IMAGE& img, CV_IMAGE& outimg, int pose_x, int pose_y, int domain_x, int domain_y) {
	if (outimg.channels != img.channels || outimg.rows != domain_y || outimg.cols != domain_x) std::cerr << "outimg wrong size !!!" << std::endl;
	outimg.data = new uchar[outimg.channels * outimg.cols * outimg.rows];
	int i = 0;
	for (int y = pose_y; y < pose_y + outimg.rows; ++y) {
		for (int x = pose_x; x < pose_x + outimg.cols; ++x) {
			for (int j = 0; j < img.channels; ++j) {
				outimg.data[i * outimg.channels + j] = img.data[((y - 1) * img.cols + x - 1) * img.channels + j];
			}
			i++;
		}
	}
}

void _get_domian(CV_IMAGE& img, CV_IMAGE& outimg, int pose_x, int pose_y, int domain_x, int domain_y) {
	int i = 0;
	for (int y = pose_y; y < pose_y + outimg.rows; ++y) {
		for (int x = pose_x; x < pose_x + outimg.cols; ++x) {
			for (int j = 0; j < img.channels; ++j) {
				outimg.data[i * outimg.channels + j] = img.data[((y - 1) * img.cols + x - 1) * img.channels + j];
			}
			i++;
		}
	}
}

void process::stick_on_domain(CV_IMAGE& stickimg, CV_IMAGE& outimg, int pose_x, int pose_y) {
	int i = 0;
	for (int y = pose_y; y < pose_y + stickimg.rows; ++y) {
		for (int x = pose_x; x < pose_x + stickimg.cols; ++x) {
			for (int j = 0; j < stickimg.channels; ++j) {
				outimg.data[((y - 1) * outimg.cols + x - 1) * outimg.channels + j] = stickimg.data[i * stickimg.channels + j];
			}
			i++;
		}
	}
}

void process::loc_hist_equa(CV_IMAGE& img, CV_IMAGE& outimg, int domain_size) {
	CV_IMAGE temp_stick_img(domain_size, domain_size, img.channels);
	CV_IMAGE temp_img(temp_stick_img);
	for (int y = 1; y < img.rows - domain_size + 2; ++y) {
		for (int x = 1; x < img.cols - domain_size + 2; ++x) {
			_get_domian(img, temp_img, x, y, domain_size, domain_size);
			process::hist_equa(temp_img, temp_stick_img);
			process::stick_on_domain(temp_stick_img, outimg, x, y);
		}
	}
}

double* process::mean_grey_value(CV_IMAGE& img) {
	int size = img.cols * img.rows;
	double* mean_value = new double[img.channels]();
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < img.channels; ++j) {
			mean_value[j] += img.data[i * img.channels + j];
		}
	}
	for (int j = 0; j < img.channels; ++j) {
		mean_value[j] = mean_value[j] / size;
	}
	return mean_value;
}

double* process::variance(CV_IMAGE& img) {
	double* vari = new double[img.channels]();
	double* mean_value = process::mean_grey_value(img);
	for (int i = 0; i < img.cols * img.rows; ++i) {
		for (int j = 0; j < img.channels; ++j) {
			vari[j] += std::pow(img.data[i * img.channels + j] - mean_value[j], 2);
		}
	}
	delete mean_value;
	for (int j = 0; j < img.channels; ++j) {
		vari[j] = vari[j] / img.cols / img.rows;
	}
	return vari;
}
void process::add_edge(CV_IMAGE& img, CV_IMAGE& outimg, int size) {
	if (outimg.cols != img.cols + 2 * size || outimg.rows != img.rows + 2 * size || outimg.channels != img.channels) std::cerr << "outimg wrong size !!!" << std::endl;
	process::stick_on_domain(img, outimg, size + 1, size + 1);
}

void process::enhance(CV_IMAGE& img, CV_IMAGE& outimg, int domain_size, double E, double mean_value_scal, double low_vari_scal, double high_vari_scal) {
	int size = img.cols * img.rows;
	int edge_size = (domain_size - 1) / 2;
	CV_IMAGE temp_out_img(img.cols + 2 * edge_size, img.rows + 2 * edge_size, img.channels);
	CV_IMAGE temp_domain_img(domain_size, domain_size, img.channels);

	process::add_edge(img, temp_out_img, edge_size);
	double* var = process::variance(img), * temp_var;
	double* mean = process::mean_grey_value(img), * temp_mean;

	for (int y = 1; y < temp_out_img.rows - domain_size + 2; ++y) {
		for (int x = 1; x < temp_out_img.cols - domain_size + 2; ++x) {
			_get_domian(temp_out_img, temp_domain_img, x, y, domain_size, domain_size);
			temp_mean = process::mean_grey_value(temp_domain_img);
			temp_var = process::variance(temp_domain_img);
			for (int j = 0; j < img.channels; ++j) {
				if (temp_mean[j] > mean_value_scal* mean[j] || temp_var[j] > high_vari_scal* var[j] || temp_var[j] < low_vari_scal * var[j]) {
					outimg.data[((y - 1) * outimg.cols + x - 1) * outimg.channels + j] = img.data[((y - 1) * img.cols + x - 1) * img.channels + j];
				}
				else {
					outimg.data[((y - 1) * outimg.cols + x - 1) * outimg.channels + j] = E * img.data[((y - 1) * img.cols + x - 1) * img.channels + j];
				}
			}
			delete temp_mean;
			delete temp_var;
		}
	}
	delete var;
	delete mean;
}

void process::diff(CV_IMAGE& img, CV_IMAGE& outimg) {
	for (int i = 0; i < img.channels * img.cols * img.rows; ++i) {
		if (outimg.data[i] == img.data[i]) outimg.data[i] = 0;
		else outimg.data[i] = 255;
	}
}

void process::flip_img(CV_IMAGE& img) {
	int size = img.cols * img.rows;
	uchar* temp_data = new uchar[size * img.channels];
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < img.channels; ++j) {
			temp_data[i * img.channels + j] = img.data[(size - i - 1) * img.channels + j];
		}
	}
	delete img.data;
	img.data = temp_data;
}

void process::flip_core(CONV_CORE& core) {
	int size = core.cols * core.rows;
	double* temp_data = new double[size * core.channels];
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < core.channels; ++j) {
			temp_data[i * core.channels + j] = core.data[(size - i - 1) * core.channels + j];
		}
	}
	delete core.data;
	core.data = temp_data;
}

double* _multplay(CV_IMAGE& temp_domain_img, CONV_CORE& core) {
	double* temp_rgb_value = new double[core.channels]();
	int size = temp_domain_img.cols * temp_domain_img.rows;
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < core.channels; ++j) {
			temp_rgb_value[j] += core.data[i * core.channels + j] * temp_domain_img.data[i * temp_domain_img.channels + j];
		}
	}
	for (int j = 0; j < core.channels; ++j) {
		temp_rgb_value[j] = temp_rgb_value[j] > 255 ? 255 : temp_rgb_value[j];
		temp_rgb_value[j] = temp_rgb_value[j] > 0 ? temp_rgb_value[j] : 0;
	}
	return temp_rgb_value;

}

void process::convolute(CV_IMAGE& img, CV_IMAGE& outimg, CONV_CORE& core) {
	//仅处理方形核
	int domain_size = core.cols;
	int edge_size = (domain_size - 1) / 2;
	CV_IMAGE temp_domain_img(core);
	CV_IMAGE temp_out_img(img.cols + 2 * edge_size, img.rows + 2 * edge_size, img.channels);
	process::add_edge(img, temp_out_img, edge_size);
	double* point_rgb_value;
	for (int y = 1; y < temp_out_img.rows - domain_size + 2; ++y) {
		for (int x = 1; x < temp_out_img.cols - domain_size + 2; ++x) {
			_get_domian(temp_out_img, temp_domain_img, x, y, domain_size, domain_size);
			point_rgb_value = _multplay(temp_domain_img, core);
			for (int j = 0; j < core.channels; ++j) {
				outimg.data[((y - 1) * outimg.cols + x - 1) * outimg.channels + j] = point_rgb_value[j];
			}
			delete point_rgb_value;
		}
	}
}

void process::addimg(CV_IMAGE& img, CV_IMAGE& addimg, CV_IMAGE& outimg, double k) {
	int size = img.cols * img.cols * img.channels;
	double temp;
	for (int i = 0; i < size; ++i) {
		temp = img.data[i] + k * addimg.data[i];
		outimg.data[i] = temp > 255 ? 255 : temp;
	}
}

void process::minusimg(CV_IMAGE& img, CV_IMAGE& minus_img, CV_IMAGE& outimg) {
	int size = img.rows * img.cols;
	uchar temp;
	for (int i = 0; i < size * img.channels; ++i) {
		temp = img.data[i] - minus_img.data[i];
		outimg.data[i] = temp > 0 ? temp : 0;
	}
}

void process::sebol(CV_IMAGE& img, CV_IMAGE& outimg, CONV_CORE& sebol_core_x, CONV_CORE& sebol_core_y) {
	int edge_size = (sebol_core_x.cols - 1) / 2;
	CV_IMAGE temp_img(sebol_core_x);
	CV_IMAGE temp_out_img(img.cols + 2 * edge_size, img.rows + 2 * edge_size, img.channels);
	process::add_edge(img, temp_out_img, edge_size);
	double* temp_grad_x, * temp_grad_y;
	for (int y = 1; y < temp_out_img.rows - sebol_core_x.rows + 2; ++y) {
		for (int x = 1; x < temp_out_img.cols - sebol_core_x.cols + 2; ++x) {
			_get_domian(temp_out_img, temp_img, x, y, temp_img.cols, temp_img.rows);
			temp_grad_x = _multplay(temp_img, sebol_core_x);
			temp_grad_y = _multplay(temp_img, sebol_core_y);
			for (int j = 0; j < img.channels; ++j) {
				outimg.data[((y - 1) * outimg.cols + x - 1) * outimg.channels + j] = std::abs(temp_grad_x[j]) + std::abs(temp_grad_y[j]);
			}
			delete temp_grad_x;
			delete temp_grad_y;
		}
	}
}

void _fft_shift(CV_IMAGE& img) {
	CV_IMAGE img1(img, 1, 1, img.cols / 2, img.rows / 2), img2(img, img.cols / 2 + 1, 1, img.cols / 2, img.rows / 2);
	CV_IMAGE img3(img, 1, 1 + img.rows / 2, img.cols / 2, img.rows / 2), img4(img, img.cols / 2 + 1, img.rows / 2 + 1, img.cols / 2, img.rows / 2);
	process::stick_on_domain(img1, img, img.cols / 2 + 1, img.rows / 2 + 1);
	process::stick_on_domain(img2, img, 1, img.rows / 2 + 1);
	process::stick_on_domain(img3, img, 1 + img.cols / 2, 1);
	process::stick_on_domain(img4, img, 1, 1);
}

void process::fft_gray_img(CV_IMAGE& gray_img, CV_IMAGE& gray_outimg) {//, double (*f)(double)){
	fftw_complex* in, * out;
	fftw_plan plan_fftw_img;
	int size = gray_img.cols * gray_img.rows;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);

	for (int i = 0; i < size; ++i) {
		in[i][0] = gray_img.data[i];
		in[i][1] = 0;
	}

	plan_fftw_img = fftw_plan_dft_2d(gray_outimg.rows, gray_outimg.cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan_fftw_img);
	double min = 0, max = 0, * temp = new double[size];

	for (int i = 0; i < size; ++i) {
		temp[i] = std::log(std::sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]));
		if (temp[i] <= min) min = temp[i];
		if (temp[i] >= max) max = temp[i];
	}
	max = max - min;

	for (int i = 0; i < size; ++i) {
		gray_outimg.data[i] = (temp[i] - min) / max * 255;
	}
	_fft_shift(gray_outimg);
	delete[] temp;
	fftw_destroy_plan(plan_fftw_img);
	fftw_free(in);
	fftw_free(out);
}

void process::transfer_gray_value(CV_IMAGE& img, double(*f)(double)) {
	for (int i = 0; i < img.cols * img.rows; ++i) {
		img.data[i] = f(img.data[i]);
	}
}

double CORE::gauss(int x, int y, double sigma) {
	return 1.0 / (std::sqrt(2 * M_PI) * sigma * sigma) * std::exp(-(x * x + y * y) / (2 * sigma * sigma));
}

double* CORE::gauss_core_data(int size, int channels, double siamg) {
	int half_size = (size - 1) / 2 + 1;
	double sum = 0, temp;
	double* data = new double[size * size * channels];
	for (int y = 1; y < size + 1; ++y) {
		for (int x = 1; x < size + 1; ++x) {
			temp = gauss(x - half_size, y - half_size, siamg);
			sum += temp;
			for (int j = 0; j < channels; ++j) {
				data[((y - 1) * size + x - 1) * channels + j] = temp;
			}
		}
	}
	for (int j = 0; j < size * size * channels; ++j) {
		data[j] = data[j] / sum;
	}
	return data;
}
