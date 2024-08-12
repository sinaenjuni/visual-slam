#include "iostream"
#include "opencv2/opencv.hpp"
#include "file.h"

struct Extractor_data{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Point2f> good_pt;
    // std::array<cv::Point2f, 3000> good_pt;
};

class Extractor
{
private:
    cv::Mat gray;
    Extractor_data curr;
    Extractor_data prev;
    
    cv::Ptr<cv::ORB> orb;
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;

public:
    Extractor();
    int compute(cv::Mat &frame);
    ~Extractor();
};

Extractor::Extractor()
{
    orb = cv::ORB::create(500, 1.2F, 8, 31, 0, 3, cv::ORB::FAST_SCORE, 31, 20);
    matcher = cv::BFMatcher(cv::NORM_HAMMING2, false);
}
int Extractor::compute(cv::Mat &frame)
{
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    // cv::GaussianBlur(gray, gray, cv::Size(3,3), 3, 3);

    orb->detectAndCompute(gray, cv::noArray(), curr.keypoints, curr.descriptors);
    if(prev.keypoints.size() <= 0){
        prev = curr;
        return 1;
    }

    matcher.match(curr.descriptors, prev.descriptors, matches);
    PRINT(curr.keypoints.size());
    for(int i=0; i<matches.size(); ++i){
        cv::Point2f p0 = curr.keypoints[matches[i].queryIdx].pt;
        cv::Point2f p1 = prev.keypoints[matches[i].trainIdx].pt;
        if(cv::norm(p0-p1) > 20){
            continue;
        }
        // cv::circle(frame, p1, 10, cv::Scalar(255,0,255), 2, cv::LINE_AA);
        cv::line(frame, p0, p1, cv::Scalar(255,0,0), 1, cv::LINE_AA);
    }

    prev = curr;
    return 2;
}
Extractor::~Extractor()
{
}



int main(int, char**){
    cv::Mat frame, gray;
    cv::VideoCapture cap("../data/test.mp4");
    // int HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    // int WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    // PRINT(HEIGHT, WIDTH);

	if (!cap.isOpened())
	{
		printf("Can't open the camera");
		return -1;
	}

    Extractor extrator;
    int ret = 0;

    while (1)
    {
        cap >> frame;
        cv::resize(frame, frame, cv::Size(960, 540));

        ret = extrator.compute(frame);
        PRINT(ret);
        if(ret == 1){
            continue;
        }


        cv::imshow("frame", frame);
        if(cv::waitKey(1) == 27) // ESC
            break;
    }
    

    printf("@@@Hello, from visual_slam!\n");
}
