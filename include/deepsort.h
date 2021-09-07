#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "featuretensor.h"
#include "tracker.h"
#include "datatype.h"
#include <vector>

using std::vector;
using nvinfer1::ILogger;

class DeepSort {
public:    
    DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID, ILogger* gLogger);
    ~DeepSort();

public:
    void sort(cv::Mat& frame, vector<DetectBox>& dets);

private:
    void sort(cv::Mat& frame, DETECTIONS& detections);
    void sort(cv::Mat& frame, DETECTIONSV2& detectionsv2);    
    void sort(vector<DetectBox>& dets);
    void sort(DETECTIONS& detections);
    void init();

private:
    std::string enginePath;
    int batchSize;
    int featureDim;
    cv::Size imgShape;
    float confThres;
    float nmsThres;
    int maxBudget;
    float maxCosineDist;

private:
    vector<RESULT_DATA> result;
    vector<std::pair<CLSCONF, DETECTBOX>> results;
    tracker* objTracker;
    FeatureTensor* featureExtractor;
    ILogger* gLogger;
    int gpuID;
};

#endif  //deepsort.h
