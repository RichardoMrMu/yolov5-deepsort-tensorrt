
#pragma once 
#include "deepsort.h"
#ifdef __cplusplus
extern "C" 
{
#endif 
 
void * yolov5_trt_create(const char * engine_name);
 
int yolov5_trt_detect(void *h, cv::Mat &img, float threshold,std::vector<DetectBox>& det);
 
void yolov5_trt_destroy(void *h);
 
#ifdef __cplusplus
}
#endif 


