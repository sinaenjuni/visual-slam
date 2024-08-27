#include <iostream>

#include "file.h"
#include "viewer.h"
#include "frame.h"

#include "pangolin/pangolin.h"
#include "sophus/se3.hpp"
#include "Eigen/eigen"
#include "opencv2/opencv.hpp"
#include <opencv2/core/eigen.hpp>

int main(int, char**){
    // cv::VideoCapture cap("../data/00/image_0/%06d.png");

    fs::path p("../data/00/image_0/");
    fs::directory_iterator iter(p);

    std::vector<std::string> file_list;
    for(const fs::path& file : iter)
    {
        if(file.has_extension() && check_file_extensions(file))
        {
            file_list.push_back(fs::absolute(file));
        }
    }
    std::sort(file_list.begin(), file_list.end(), compare);

    // std::vector<cv::Mat> imgs;
    // for(auto& t : file_list){
        // PRINT(t);
        // cv::Mat img = cv::imread(t, cv::IMREAD_COLOR);
        // imgs.push_back(img);
    // }
    
    cv::Mat K = (cv::Mat_<double>(3,3) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
                                            0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02,
                                            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);
    cv::Mat Kinv = K.inv();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        KinvE(Kinv.ptr<double>(), Kinv.rows, Kinv.cols);

    int WIDTH = 1241;
    int HEIGHT = 376;

    Viewer viewer(WIDTH, HEIGHT, KinvE);


	// if (!cap.isOpened())
	// {
	// 	printf("Can't open the camera");
	// 	return -1;
	// }

    cv::Ptr<cv::ORB> feature_extractor = cv::ORB::create(3000, 1.2F, 8, 31, 0, 3, cv::ORB::FAST_SCORE, 31, 25);
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, false);


    cv::Mat image, frame_curr, frame_prev;
    std::vector<Frame<cv::ORB>> frames;
    bool is_init_pose = true;
    int ind = 0;
    while (1)
    {
        // cap >> image;
        image = cv::imread(file_list[ind], cv::IMREAD_UNCHANGED);
        // frame_curr = cv::imread(file_list[ind], cv::IMREAD_UNCHANGED);
        // cv::resize(image, image, cv::Size(WIDTH, HEIGHT));

        Frame<cv::ORB> frame(image, feature_extractor, matcher, K);
        
        if(ind < 1){
            // frame_curr.copyTo(frame_prev);
            PRINT("Just one image");
            frames.push_back(frame);
            ++ind;
            continue;
        }

        frame.calcPoseWithF(frames.back());
        // frame.calcObjPoints(frames.back());
        frames.push_back(frame);

        viewer.draw(frames);

        cv::imshow("frame", image);
        // cv::imshow("frame", frame_curr);
        // if(cv::waitKey() == 27 || pangolin::ShouldQuit()) // ESC
        // if(cv::waitKey() == 27) // ESC
            // break;

        int key = cv::waitKey(33);
        if (key == 32) key = cv::waitKey(); // Space
        if (key == 27) break;               // ESC
        // frame_curr.copyTo(frame_prev);
        ++ind;
    }
    

    printf("@@@Hello, from visual_slam!\n");
}
