#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "deepsort.h"
#include "logging.h"
#include <ctime>

#include "time.h"

using std::vector;

static Logger gLogger;

void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 1);
        std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0,255,0));
    }
    cv::imshow("img", temp);
    cv::waitKey(1);
}

class Tester {
public:
    Tester(string modelPath) {
        allDetections.clear();
        out.clear();
        DS = new DeepSort(modelPath, 128, 256, 0, &gLogger);
    }
    ~Tester() {
    }

public:

    void split(const std::string& s, vector<std::string>& token, char delim=' ') {
        token.clear();
        auto string_find_first_not = [s, delim](size_t pos = 0) -> size_t {
            for (size_t i = pos; i < s.size(); ++i)
                if (s[i] != delim) return i;
            return string::npos;
        };
        size_t lastPos = string_find_first_not(0);
        size_t pos = s.find(delim, lastPos);
        while (lastPos != string::npos) {
            token.emplace_back(s.substr(lastPos, pos-lastPos));
            lastPos = string_find_first_not(pos);
            pos = s.find(delim, lastPos);
        }
    }

    void loadDetections(std::string txtPath) {
        //fstream f(filePath, ios::in);
        this->txtPath = txtPath;
        ifstream inFile;
        inFile.open(txtPath, ios::binary);
        std::string temp;
        vector<std::string> token;
        clock_t start, finish;
        double  duration; 
        start = clock(); 
        while (getline(inFile, temp)) {
            // std::cout << temp << std::endl;
            split(temp, token, ' ');
            int frame = atoi(token[0].c_str());
            if (frame==4){
            	finish = clock(); 
            duration = (double)(finish - start) / CLOCKS_PER_SEC;    
            printf( "%f seconds\n", duration );
            }
            int c     = atoi(token[1].c_str());
            int x     = atoi(token[2].c_str());
            int y     = atoi(token[3].c_str());
            int w     = atoi(token[4].c_str());
            int h     = atoi(token[5].c_str());
            float con= atof(token[6].c_str());     
            while (allDetections.size() <= frame) {
            	
                vector<DetectBox> t;
                allDetections.push_back(t);
            }
            DetectBox dd(x-w/2, y-h/2, x+w/2, y+h/2, con, c);
            allDetections[frame].push_back(dd);
        }
        allDetections.pop_back();
    }

    void run() {
        cv::namedWindow("DeepSortTest");
        int i = 1;
        cv::Mat whiteBoard(1080, 1920, CV_8UC3, cv::Scalar::all(0));
        for (vector<DetectBox> d : allDetections) {
            cv::Mat img_rgb;
            cv::cvtColor(whiteBoard, img_rgb, cv::COLOR_BGR2RGB);
            clock_t start, finish;
            double  duration;     
            start = clock(); 
            DS->sort(img_rgb, d);
            finish = clock(); 
            duration = (double)(finish - start) / CLOCKS_PER_SEC;    
            printf( "%f seconds\n", duration );
	
            showDetection(whiteBoard, d);
        }
        cv::destroyAllWindows();
    }

private:
    vector<vector<DetectBox>> allDetections;
    vector<DetectBox> out;
    std::string txtPath;
    DeepSort* DS;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "./demo [input model path] [input txt path]" << std::endl;
        return -1;
    }
    Tester* test = new Tester(argv[1]);
    test->loadDetections(argv[2]);
    test->run();
}
