#include "tools.h"

#define WINDOW_NAME "WINDOW"
#define	IMAGE_SIZE Size(40, 40) 
#define VIDEO_MAX_WIDTH 640
#define VIDEO_MAX_HEIGHT 480

using namespace cv;
using namespace cv::ml;
using namespace std;

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		cout << "Wrong arguments" << endl;
		cout << "Usage: exe videoFilePath svmFilePath" << endl;
		return -1;
	}

	//Get video path
	string video_file_path(argv[1]);

	//Get image size
	const Size & size = IMAGE_SIZE;

	//Get trained svm file
	string svm_file_path(argv[2]);

	char key = 27;
	Mat img, draw;
	Ptr<SVM> svm;
	HOGDescriptor hog;
	hog.winSize = size;
	VideoCapture video;
	vector< Rect > locations;

	// Load the trained SVM.
	svm = StatModel::load<SVM>(svm_file_path);
	// Set the trained svm to my_hog
	vector< float > hog_detector;
	get_svm_detector(svm, hog_detector);
	hog.setSVMDetector(hog_detector);

	// Open the camera.
	video.open(video_file_path);
	if (!video.isOpened())
	{
		cerr << "Unable to open the device" << endl;
		exit(-1);
	}

	Size videoSize = Size((int)video.get(CAP_PROP_FRAME_WIDTH), (int)video.get(CAP_PROP_FRAME_HEIGHT));
	bool enableResize = videoSize.width > VIDEO_MAX_WIDTH || videoSize.height > VIDEO_MAX_HEIGHT;

	int num_of_vehicles = 0;

	bool end_of_process = false;
	while (!end_of_process)
	{
		video >> img;
		if (img.empty())
			break;
		else
		{
			if (enableResize)
				resize(img, img, Size(VIDEO_MAX_WIDTH, VIDEO_MAX_HEIGHT));
		}

		draw = img.clone();

		locations.clear();
		hog.detectMultiScale(img, locations);
		draw_locations(draw, locations, Scalar(0, 255, 0));

		//for each(Rect r in locations) {

		//	// Center point of the vehicle
		//	Point center(r.x + r.width / 2, r.y + r.height / 2);

		//	if (abs(center.y - img.rows * 2 / 3) < 2) {
		//		++num_of_vehicles;
		//		line(draw, Point(0, img.rows * 2 / 3), Point(img.cols / 2, img.rows * 2 / 3), Scalar(0, 255, 0), 3);
		//		imshow(WINDOW_NAME, draw);
		//		waitKey(50);
		//	}
		//	else
		//		line(draw, Point(0, img.rows * 2 / 3), Point(img.cols / 2, img.rows * 2 / 3), Scalar(0, 0, 255), 3);

		//}

		//putText(draw, "Detected vehicles: " + to_string(num_of_vehicles), Point(50, 50), 1, 1, Scalar(0, 0, 255), 2);

		imshow(WINDOW_NAME, draw);
		key = (char)waitKey(10);
		if (27 == key)
			end_of_process = true;
	}

	return true;
}
