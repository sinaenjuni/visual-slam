#include <iostream>
#include "opencv2/opencv.hpp"
#include "file.h"

#include "pangolin/pangolin.h"
#include "sophus/se3.hpp"
#include "Eigen/eigen"


struct Extractor_data{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Point2f> matched_points;
};

class Extractor
{
private:
    cv::Mat prev_gray, curr_gray;
    Extractor_data curr;
    Extractor_data prev;
    
    cv::Ptr<cv::ORB> orb;
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;

    cv::Mat F, inlier;
    cv::Mat K;
public:
    Extractor(cv::Mat K);
    void computeRt(cv::Mat &prev_frame, cv::Mat &curr_frame, 
                    cv::Mat &R, cv::Mat &t);
    void getPose(cv::Mat &R, cv::Mat &t);
    void reset();
    ~Extractor();
};

Extractor::Extractor(cv::Mat K) : K(K)
{
    orb = cv::ORB::create(1000, 1.2F, 8, 31, 0, 3, cv::ORB::FAST_SCORE, 31, 25);
    matcher = cv::BFMatcher(cv::NORM_HAMMING2, false);
}
void Extractor::computeRt(cv::Mat &prev_frame, cv::Mat &curr_frame, 
                            cv::Mat &R, cv::Mat &t)
{
    cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(prev_gray, prev_gray, cv::Size(3,3), 0);
    cv::GaussianBlur(curr_gray, curr_gray, cv::Size(3,3), 0);
    reset();

    orb->detectAndCompute(prev_gray, cv::noArray(), prev.keypoints, prev.descriptors);
    orb->detectAndCompute(curr_gray, cv::noArray(), curr.keypoints, curr.descriptors);

    matcher.match(curr.descriptors, prev.descriptors, matches);
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
        return a.distance < b.distance;
        });

    for(int i=0; i<1000; ++i){
        // cv::DMatch curr_match = matches[i][0], prev_match = matches[i][1];
        // PRINT(curr_match.distance, prev_match.distance);
        // if(curr_match.distance < 0.75*prev_match.distance){
            // PRINT(curr_match.distance, prev_match.distance);
        cv::Point2f p0 = prev.keypoints[matches[i].trainIdx].pt;
        cv::Point2f p1 = curr.keypoints[matches[i].queryIdx].pt;
        // PRINT(cv::norm(p0-p1));
        // if(cv::norm(p0-p1) > 50){
        //         continue;
        // }
        prev.matched_points.push_back(p0);
        curr.matched_points.push_back(p1);
            // continue;
        // cv::circle(curr_frame, p1, 10, cv::Scalar(255,0,255), 2, cv::LINE_AA);

        // cv::line(curr_frame, p0, p1, cv::Scalar(255,0,0), 1, cv::LINE_AA);
        // }

        // cv::circle(frame, p1, 10, cv::Scalar(255,0,255), 2, cv::LINE_AA);
        // cv::line(curr_frame, p0, p1, cv::Scalar(255,0,0), 1, cv::LINE_AA);
    }
    // cv::Point2f curr_sum(0.0f, 0.0f);
    // cv::Point2f prev_sum(0.0f, 0.0f);

    // for(int i=0; i<curr.matched_points.size(); ++i){
    //     curr_sum += curr.matched_points[i];
    //     prev_sum += prev.matched_points[i];
    // }
    // curr_sum = curr_sum / static_cast<float>(curr.matched_points.size());
    // curr_sum = curr_sum / static_cast<float>(curr.matched_points.size());
    // PRINT(curr_sum);
    // sum = prev_sum/curr.matched_points.size());
    // F = cv::findFundamentalMat(curr.matched_points, prev.matched_points, inlier, cv::FM_LMEDS);
    // PRINT(cv::countNonZero(inlier));
    F = cv::findEssentialMat(curr.matched_points, prev.matched_points, K, cv::RANSAC);
    // for(int i=0; i<inlier.rows; ++i){
    //     if(inlier.at<bool>(i)){
    //         cv::Point2f p0 = curr.matched_points[i];
    //         cv::Point2f p1 = prev.matched_points[i];
    //         cv::line(curr_frame, p0, p1, cv::Scalar(255,0,0), 1, cv::LINE_AA);
    //     }
    // }
    cv::recoverPose(F, curr.matched_points, prev.matched_points, K, R, t, cv::noArray());

    // cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0,
    //                                         1, 0,  0,
    //                                         0, 0,  1);
    // cv::Mat U, D, Vt;
    // cv::SVDecomp(F, D, U, Vt);
    // // PRINT(Vt);
    // // PRINT(cv::determinant(Vt));

    // if(cv::determinant(U)<0){
    //     U *= -1.0;
    // }
    // if(cv::determinant(Vt)<0){
    //     Vt *= -1.0;
    // }
    // R = U * W * Vt;
    // if(cv::sum(R.diag())[0] < 0){
    //     R = U * W.t() * Vt;
    // }
    // // PRINT(cv::determinant(R));
    // t = U.col(2);
}

void Extractor::getPose(cv::Mat &R, cv::Mat &t){
    F = cv::findFundamentalMat(curr.matched_points, prev.matched_points, inlier);
    cv::recoverPose(F, curr.matched_points, prev.matched_points, K, R, t);
}

void Extractor::reset(){
    curr.matched_points.clear();
    prev.matched_points.clear();
}


Extractor::~Extractor()
{
}

int main(int, char**){
    fs::path p("../data/00/image_0/");
    fs::directory_iterator iter(p);

    std::vector<std::string> file_list;
    for(const fs::path& file : iter){
        if(file.has_extension() && check_file_extensions(file)){
            file_list.push_back(fs::absolute(file));
        }
    }
    
    std::sort(file_list.begin(), file_list.end(), compare);
    std::vector<cv::Mat> imgs;
    for(auto& t : file_list){
        // PRINT(t);
        cv::Mat img = cv::imread(t, cv::IMREAD_COLOR);
        imgs.push_back(img);
    }
    
    cv::Mat K = (cv::Mat_<double>(3,3) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
                                            0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02,
                                            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);
    // cv::Mat Kinv = K.inv();

    int WIDTH=imgs[0].size().width, HEIGHT=imgs[0].size().height;
    cv::Mat prev_frame, curr_frame;
    cv::Mat R, t;
    // cv::VideoCapture cap("../data/test.mp4");
    // // int HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    // // int WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    // // PRINT(HEIGHT, WIDTH);
    // cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    // K.at<double>(0, 0) = 1000;
    // K.at<double>(1, 1) = 1000;

    // K.at<double>(0, 2) = WIDTH/2;
    // K.at<double>(1, 2) = HEIGHT/2;
    // // cv::invert(K, Kinv);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        Kinv(K.ptr<double>(), K.rows, K.cols);
    PRINT(Kinv);
    Kinv = Kinv.inverse();
    PRINT(Kinv);

    // // PRINT(K);


    pangolin::CreateWindowAndBind("Main",640, 480);
    glEnable(GL_DEPTH_TEST);

    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480, 420,420, 320,240, 0.2,10000),
        pangolin::ModelViewLookAt(-10, 10, -10, 0,0,0, 0.0,-1.0, 0.0)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
            .SetHandler(&handler);


	// if (!cap.isOpened())
	// {
	// 	printf("Can't open the camera");
	// 	return -1;
	// }

    Extractor extrator(K);
    // int ret = 0;

    std::vector<Sophus::SE3d> Ts;
    // Sophus::SE3d initT(Eigen::Matrix4d::Identity());
    // PRINT(initT.matrix());
    // PRINT(Eigen::Matrix4d::Identity());
    // Ts.push_back(initT);
    // PRINT(Ts.back().matrix());
    
    // cv::Mat initP = cv::Mat::eye(cv::Size(4,4), CV_64F);
    // initP = initP;

    int ind = 0;
    while (1)
    {
        // cap >> curr_frame;
        curr_frame = imgs[ind];
        // cv::resize(curr_frame, curr_frame, cv::Size(WIDTH, HEIGHT));
        
        if(prev_frame.empty()){
            curr_frame.copyTo(prev_frame);
            ++ind;
            PRINT("COPY");
        }

        extrator.computeRt(prev_frame, curr_frame, R, t);
        // cv::Mat P;
        // cv::hconcat(R, t, P);
        // P = K * P;

        // PRINT(K * P);
        PRINT(R, t);
        // PRINT(t);
        // cv::triangulatePoints()



        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        eigenR(R.ptr<double>(), R.rows, R.cols);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        eigent(t.ptr<double>(), t.rows, t.cols);

        Sophus::SE3d T(eigenR, eigent);
        // T = T.inverse();
        if(Ts.size() <= 1){
            Ts.push_back(T);
            continue;
        }
        Ts.push_back(T * Ts.back());
        // Ts.push_back(Ts.back() * T);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        
        pangolin::glDrawAxis(1);
        // Eigen::Matrix3f eigenR;
        // Eigen::Vector3f eigent;
        for(int i=0; i<Ts.size(); ++i){
            pangolin::glDrawFrustum<double>(Kinv.matrix(), WIDTH, HEIGHT, Ts[i].inverse().matrix(), 1);
        }
        // PRINT(Ts.back().translation().x());
        // Sophus::SE3d t = Ts.back();
        // PRINT(t.transX);
        // PRINT(t.transY);
        // PRINT(t.transZ);

        // mv.setRotationMatrix(Eigen::Matrix3d::Identity());
        // mv = Ts.back().setRotationMatrix(Sophus::Matrix3d::Identity());
        // PRINT(mv.matrix());

        // OpenGL 호환 행렬로 변환
        // pangolin::OpenGlMatrix m(mv.matrix());

        // 모델 뷰 행렬을 설정하여 시점을 새로운 위치로 이동
        // s_cam.SetModelViewMatrix(m);

        // s_cam.GetModelViewMatrix();
        // PRINT(s_cam.GetModelViewMatrix());
        // s_cam.SetModelViewMatrix()

        pangolin::FinishFrame();

        cv::imshow("frame", curr_frame);
        // if(cv::waitKey() == 27 || pangolin::ShouldQuit()) // ESC
        if(cv::waitKey() == 27) // ESC
            break;
            
        curr_frame.copyTo(prev_frame);
        ++ind;
    }
    

    printf("@@@Hello, from visual_slam!\n");
}
