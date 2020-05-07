#ifndef TOOLS_H
#define TOOLS_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>

// includes for file_exists and files_in_directory functions
#ifndef __linux
#include <io.h> 
#define access _access_s
#else
#include <unistd.h>
#include <memory>
#endif

using namespace cv;
using namespace cv::ml;
using namespace std;

bool file_exists(const string &file);
void load_images(string directory, vector<Mat>& image_list);
vector<string> files_in_directory(string directory);

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);
void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size);
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size);
void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels);
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);
void test_it(const Size & size);

#endif
