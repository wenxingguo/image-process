#pragma once
#ifndef IMG_H
#define IMG_H

#define _USE_MATH_DEFINES
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "FL/Fl.H"
#include "fftw3.h"
#include "FL/Fl_Window.H"
#include "FL/Fl_Box.H"
#include "FL/Fl_RGB_Image.H"
#include "FL/fl_ask.H"
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

class CONV_CORE;

class CV_IMAGE {
private:
	void convert_CV_img(cv::Mat& img);
public:

	CV_IMAGE();
	CV_IMAGE(char* filename, int flag = 1);
	CV_IMAGE(cv::Mat& img);
	CV_IMAGE(CV_IMAGE& img);
	CV_IMAGE(int cols, int rows, int channels);
	CV_IMAGE(CV_IMAGE& img, int pose_x, int pose_y, int domain_x, int domain_y);
	CV_IMAGE(CONV_CORE& core);
	~CV_IMAGE();

	enum color_channels {
		Red = 0,
		Green = 1,
		Blue = 2,
		Alpha = 3
	};

public:
	int rows;
	int cols;
	int channels;
	uchar* data;
public:
	static std::vector<CV_IMAGE*> img_show; //静态成员只声明

	void add_show_list();
	static int imshow();
	void imgsave(const std::string filename);

};

class CONV_CORE {

public:
	double* data;
	int channels;
	int rows;
	int cols;
	CONV_CORE(int size, int in_channels);
	CONV_CORE(double* in_data, int size, int in_channels);
	CONV_CORE(CONV_CORE& r_core, CONV_CORE& g_core, CONV_CORE& b_core);
	CONV_CORE(CONV_CORE& r_core, CONV_CORE& g_core, CONV_CORE& b_core, double in_scal);
	~CONV_CORE();

};

namespace process {

	void reverse(CV_IMAGE& img);

	void log_convert(CV_IMAGE& img, int c);

	void gamma_convert(CV_IMAGE& img, double c, double gamma);

	void get_color_channel(CV_IMAGE& img, CV_IMAGE& outimg, CV_IMAGE::color_channels color);

	void rgb2image(CV_IMAGE& rimg, CV_IMAGE& gimg, CV_IMAGE& bimg, CV_IMAGE& outimg);

	void rgb2gray(CV_IMAGE& img, CV_IMAGE& outimg);

	int* hist(CV_IMAGE& img);

	void hist_equa(CV_IMAGE& img, CV_IMAGE& outimg);

	void hist_spec(CV_IMAGE& img, CV_IMAGE& outimage, int* spec_hist_array);

	enum prob_func_type {
		PROB_DENSITY,
		PROB_DISTRIB
	};

	int* gen_spec_hist(CV_IMAGE& img, double (*f)(int), process::prob_func_type f_type);

	void get_domian(CV_IMAGE& img, CV_IMAGE& outimg, int pose_x, int pose_y, int domain_x, int domain_y);

	void stick_on_domain(CV_IMAGE& stickimg, CV_IMAGE& outimg, int pose_x, int pose_y);

	void loc_hist_equa(CV_IMAGE& img, CV_IMAGE& outimg, int domain_size);

	double* mean_grey_value(CV_IMAGE& img);

	double* variance(CV_IMAGE& img);

	void add_edge(CV_IMAGE& img, CV_IMAGE& outimg, int size);

	void enhance(CV_IMAGE& img, CV_IMAGE& outimg, int domain_size, double E, double mean_value_scal, double low_vari_scla, double high_vari_scal);

	void diff(CV_IMAGE& img, CV_IMAGE& outimg);

	void flip_img(CV_IMAGE& img);

	void flip_core(CONV_CORE& core);

	void convolute(CV_IMAGE& img, CV_IMAGE& outimg, CONV_CORE& core);

	void addimg(CV_IMAGE& img, CV_IMAGE& add_img, CV_IMAGE& outimg, double k = 1);

	void minusimg(CV_IMAGE& img, CV_IMAGE& minus_img, CV_IMAGE& outimg);

	void sebol(CV_IMAGE& img, CV_IMAGE& outimg, CONV_CORE& sebol_core_x, CONV_CORE& sebol_core_y);

	void fft_gray_img(CV_IMAGE& gray_img, CV_IMAGE& gray_outimg);//, double (*f)(double));

	double shift_log(double);

	void transfer_gray_value(CV_IMAGE& in_out_img, double(*f)(double));

};

namespace CORE {
#ifdef MEAN_CORE_9
	double* temp9 = new double[9]{ 1,1,1,1,1,1,1,1,1 };
	CONV_CORE temp_core9(temp9, 3, 1);
	CONV_CORE mean_core9(temp_core9, temp_core9, temp_core9, 1.0 / 9);
#endif

#ifdef MEAN_CORE_16
	double* temp16 = new double[9]{ 1,2,1,2,4,2,1,2,1 };
	CONV_CORE temp_core16(temp16, 3, 1);
	CONV_CORE mean_core16(temp_core16, temp_core16, temp_core16, 1.0 / 16);
#endif

#ifdef LAPLACE_4
	double* temp_laplace_4 = new double[9]{ 0,-1,0,-1, 4, -1, 0, -1, 0 };
	CONV_CORE temp_core_laplace_4(temp_laplace_4, 3, 1);
	CONV_CORE laplace_core_4(temp_core_laplace_4, temp_core_laplace_4, temp_core_laplace_4);
#endif

#ifdef LAPLACE_4_PLUS
	double* temp_laplace_4_plus = new double[9]{ 0,-1,0,-1, 5, -1, 0, -1, 0 };
	CONV_CORE temp_core_laplace_4_plus(temp_laplace_4_plus, 3, 1);
	CONV_CORE laplace_core_4_plus(temp_core_laplace_4_plus, temp_core_laplace_4_plus, temp_core_laplace_4_plus);
#endif

#ifdef LAPLACE_8
	double* temp_laplace_8 = new double[9]{ -1,-1,-1,-1, 8, -1, -1, -1, -1 };
	CONV_CORE temp_core_laplace_8(temp_laplace_8, 3, 1);
	CONV_CORE laplace_core_8(temp_core_laplace_8, temp_core_laplace_8, temp_core_laplace_8);
#endif

#ifdef LAPLACE_8_PLUS
	double* temp_laplace_8_plus = new double[9]{ -1,-1,-1,-1, 9, -1, -1, -1, -1 };
	CONV_CORE temp_core_laplace_8_plus(temp_laplace_8_plus, 3, 1);
	CONV_CORE laplace_core_8_plus(temp_core_laplace_8_plus, temp_core_laplace_8_plus, temp_core_laplace_8_plus);
#endif

	double gauss(int x, int y, double sigma);

	double* gauss_core_data(int size, int channels, double sigma);

#ifdef GAUSS_CORE_SIZE
	double* temp_gauss_data = gauss_core_data(GAUSS_CORE_SIZE, 3, 0.8);
	CONV_CORE guass_core(temp_gauss_data, GAUSS_CORE_SIZE, 3);
#endif

#ifdef SEBOL
	double* temp_sebol_x = new double[9]{ -1, 0, 1, -2, 0, 2, -1, 0 , 1 };
	double* temp_sebol_y = new double[9]{ -1, -2, -1, 0, 0, 0, -1, -2, -1 };
	CONV_CORE temp_core_sebol_x(temp_sebol_x, 3, 1);
	CONV_CORE temp_core_sebol_y(temp_sebol_y, 3, 1);
	CONV_CORE sebol_core_x(temp_core_sebol_x, temp_core_sebol_x, temp_core_sebol_x);
	CONV_CORE sebol_ceor_y(temp_core_sebol_y, temp_core_sebol_y, temp_core_sebol_y);
#endif
}

#endif
