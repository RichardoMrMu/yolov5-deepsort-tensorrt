#include "manager.hpp"
using std::vector;
using namespace cv;
static Logger gLogger;

Trtyolosort::Trtyolosort(char *yolo_engine_path,char *sort_engine_path){
	sort_engine_path_ = sort_engine_path;
	yolo_engine_path_ = yolo_engine_path;
	trt_engine = yolov5_trt_create(yolo_engine_path_);
	printf("create yolov5-trt , instance = %p\n", trt_engine);
	DS = new DeepSort(sort_engine_path_, 128, 256, 0, &gLogger);

}
void Trtyolosort::showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 1);
        //std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
		//std::string lbl = cv::format("ID:%d_C:%d", (int)box.trackID, (int)box.classID);
		std::string lbl = cv::format("ID:%d_x:%f_y:%f",(int)box.trackID,(box.x1+box.x2)/2,(box.y1+box.y2)/2);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,255,0));
    }
    cv::imshow("img", temp);
    cv::waitKey(1);
}
int Trtyolosort::TrtDetect(cv::Mat &frame,float &conf_thresh,std::vector<DetectBox> &det){
	// yolo detect
	auto ret = yolov5_trt_detect(trt_engine, frame, conf_thresh,det);
	DS->sort(frame,det);
	//showDetection(frame,det);
	return 1 ;
	
}
