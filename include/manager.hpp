#ifndef _MANAGER_H
#define _MANAGER_H

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "deepsort.h"
#include "logging.h"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "time.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "yolov5_lib.h"
#include "deepsort.h"

using std::vector;
using namespace cv;
//static Logger gLogger;

class Trtyolosort{
public:
	// init 
	Trtyolosort(char *yolo_engine_path,char *sort_engine_path);
	// detect and show
	int TrtDetect(cv::Mat &frame,float &conf_thresh,std::vector<DetectBox> &det);
	void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes);

private:
	char* yolo_engine_path_ = NULL;
	char* sort_engine_path_ = NULL;
    void *trt_engine = NULL;
    // deepsort parms
    DeepSort* DS;
    std::vector<DetectBox> t;
    //std::vector<DetectBox> det;
};
#endif  // _MANAGER_H

