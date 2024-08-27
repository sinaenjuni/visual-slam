// #include <iostream>
#include <fstream>
#include <random>

#include "file.h"
#include "Eigen/eigen"
#include "Opencv2/opencv.hpp"
#include "pangolin/pangolin.h"
#include "sophus/se3.hpp"



int main(){
    std::ifstream file("../data/output.txt");
    if (!file.is_open()) {
        std::cerr << "Unable to open file for writing: " << std::endl;
    }

    std::string line;
    std::getline(file, line);
    if(line.find("K:") == std::string::npos){
        std::cerr << "K keyword is not found. " << std::endl;
    }

    Eigen::Matrix3d Keigen;
    cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
    for(int i=0; i<3; ++i)
    {
        std::getline(file, line);
        std::istringstream iss(line);
        double x, y, z;
        iss >> x >> y >> z;
        Keigen.row(i) << x, y, z;
        K.at<double>(i, 0) = x;
        K.at<double>(i, 1) = y;
        K.at<double>(i, 2) = z;
    }
    cv::Mat Kinv = K.inv();
    PRINT(K);
    PRINT(Kinv);
    PRINT(Keigen);

    // std::vector<Eigen::Matrix4d> Ps;
    // Eigen::Matrix4d P;
    // while (std::getline(file, line)) {
    //     PRINT(line);
    //     if(line.find("IMG:") != std::string::npos)
    //     {
    //         for(int i=0; i<4; ++i)
    //         {   
    //             std::getline(file, line);
    //             std::istringstream iss(line);
    //             double x1, x2, x3, x4;
    //             iss >> x1 >> x2 >> x3 >> x4;
    //             P.row(i) << x1, x2, x3, x4;
    //         }
    //         Ps.push_back(P);
    //         PRINT(P);
    //     }
    // }
    file.close();

    fs::path p("../data/calib_data/");
    fs::directory_iterator iter(p);

    std::vector<std::string> file_list;
    for(const fs::path& file : iter){
        if(file.has_extension() && check_file_extensions(file)){
            file_list.push_back(fs::absolute(file));
        }
    }
    std::sort(file_list.begin(), file_list.end(), compare);
    std::vector<cv::Mat> imgs;
    for(std::string& t : file_list){
        // PRINT(t);
        cv::Mat img = cv::imread(t, cv::IMREAD_COLOR);
        imgs.push_back(img);
    }
    int WIDTH = imgs[0].cols; 
    int HEIGHT = imgs[0].rows;
    const int N = imgs.size();

    cv::Mat corners0, corners1;
    cv::findChessboardCorners(imgs[0], cv::Size(7,9), corners0);
    cv::findChessboardCorners(imgs[13], cv::Size(7,9), corners1);

    // cv::drawChessboardCorners(imgs[0], cv::Size(7,9), corners0, true);
    // cv::drawChessboardCorners(imgs[1], cv::Size(7,9), corners1, true);

    // cv::Ptr<cv::ORB> feature_detector = cv::ORB::create();

    // std::vector<cv::KeyPoint> kpts0, kpts1;
    // cv::Mat desk0, desk1;
    // feature_detector->detectAndCompute(imgs[0], cv::noArray(), kpts0, desk0);
    // feature_detector->detectAndCompute(imgs[1], cv::noArray(), kpts1, desk1);

    // cv::BFMatcher matcher(cv::NORM_HAMMING);
    // std::vector<cv::DMatch> matches;
    // matcher.match(desk0, desk1, matches);

    cv::Mat R0;
    // R0.eye(3);


    cv::Mat F = cv::findFundamentalMat(corners0, corners1, cv::noArray());
    cv::Mat E = K.t() * F * K;
    cv::Mat R1, t1, P1;
    cv::recoverPose(E, corners0, corners1, K, R1, t1);
    cv::hconcat(R1, t1, P1);
    // cv::triangulatePoints()
    
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    RE(R1.ptr<double>(), R1.rows, R1.cols);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    tE(t1.ptr<double>(), t1.rows, t1.cols);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    PE(P1.ptr<double>(), P1.rows, P1.cols);


    Sophus::SE3d T0(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    Sophus::SE3d T1(RE, tE);


    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    KinvE(Kinv.ptr<double>(), Kinv.rows, Kinv.cols);


    PRINT(KinvE);

    pangolin::CreateWindowAndBind("Main",1024,768);
    glEnable(GL_DEPTH_TEST);

    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024,768, 2000,2000, 512,389,0.1,1000),
        pangolin::ModelViewLookAt(0,-30,0.1, 0,0,0, 0.0,-1.0, 0.0)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            // .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(&handler);

    while(!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        
        pangolin::glDrawAxis(1);
        pangolin::glDrawAxis(T0.matrix(), 1);
        pangolin::glDrawAxis(T1.matrix(), 1);

        pangolin::glDrawFrustum<double>(KinvE.matrix(), WIDTH, HEIGHT,
                                T0.matrix(), 1);

        pangolin::glDrawFrustum<double>(KinvE.matrix(), WIDTH, HEIGHT,
                                T1.matrix(), 1);

        pangolin::FinishFrame();
    }
    
    // // Select random 8-points that matched
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<int> dis(0, matches.size());

    // Eigen::MatrixXd p0_sel(8, 2), p0_sel_homo(8, 3);
    // Eigen::MatrixXd p1_sel(8, 2), p1_sel_homo(8, 3);
    // std::vector<cv::Point2d> p0_cv, p1_cv;
    // for(int i=0; i<8; ++i)
    // {   
    //     const int ind = dis(gen);
    //     p0_sel.row(i) << kpts0[ind].pt.x, kpts0[ind].pt.y;
    //     p1_sel.row(i) << kpts1[ind].pt.x, kpts1[ind].pt.y;

    //     p0_sel_homo.row(i) << kpts0[ind].pt.x, kpts0[ind].pt.y, 1;
    //     p1_sel_homo.row(i) << kpts1[ind].pt.x, kpts1[ind].pt.y, 1;

    //     p0_cv.push_back(kpts0[ind].pt);
    //     p1_cv.push_back(kpts1[ind].pt);


    //     // Eigen::Vector2d p1(kpts1[ind].pt.x, kpts1[ind].pt.y);
    //     // p0_sel.push_back(p0);
    //     // p1_sel.push_back(p1);
    //     // PRINT(p0.transpose(), p1.transpose());
    // }
    // PRINT(p0_sel);
    // PRINT(p0_sel_homo);
    // PRINT("============");

    // PRINT(p1_sel);
    // PRINT("============");

    // Eigen::Vector2d p0_mean = p0_sel.colwise().mean();
    // Eigen::Vector2d p1_mean = p1_sel.colwise().mean();
    // PRINT(p0_mean);
    // PRINT(p1_mean);
    // Eigen::MatrixXd p0_dist_mean(8, 2);
    // Eigen::MatrixXd p1_dist_mean(8, 2);
    // p0_dist_mean = (p0_sel.rowwise() - p0_mean.transpose()).array().pow(2).rowwise().sum().sqrt().colwise().mean();
    // p1_dist_mean = (p1_sel.rowwise() - p1_mean.transpose()).array().pow(2).rowwise().sum().sqrt().colwise().mean();
    
    // double s0 = std::sqrt(2.0) / p0_dist_mean(0);
    // double s1 = std::sqrt(2.0) / p1_dist_mean(0);

    // PRINT("s0", s0);
    // PRINT("s1", s1);

    // Eigen::Matrix3d T0, T1;
    // T0 << s0, 0, -(s0 * p0_mean(0)),
    //       0, s0, -(s0 * p0_mean(1)),
    //       0, 0, 1;
    // T1 << s1, 0, -(s1 * p1_mean(0)),
    //       0, s1, -(s1 * p1_mean(1)),
    //       0, 0, 1;

    // PRINT(T0);
    // PRINT(T1);
    
    // p0_sel_homo = (T0 * p0_sel_homo.transpose()).transpose();
    // p1_sel_homo = (T1 * p1_sel_homo.transpose()).transpose();

    // PRINT("p0_sel_homo");
    // PRINT(p0_sel_homo);

    // Eigen::MatrixXd A(8, 9); // X1*F*X0 = 0
    // for (int i = 0; i < 8; ++i) {
    //     Eigen::Vector3d p0 = T0*p0_sel_homo.row(i).transpose();
    //     Eigen::Vector3d p1 = T1*p1_sel_homo.row(i).transpose();

    //     double x0 = p0(0), y0 = p0(1);
    //     double x1 = p1(0), y1 = p1(1);
    //     A.row(i) << x1*x0, x1*y0, x1, y1*x0, y1*y0, y1, x0, y0, 1;
    // }

    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Eigen::MatrixXd V = svd.matrixV();
    // Eigen::Matrix3d F = V.col(8).reshaped(3, 3);

    // // F 행렬의 특성을 만족하도록 제약
    // svd.compute(F);
    // Eigen::VectorXd singular_values = svd.singularValues();
    // singular_values(2) = 0;
    // F = svd.matrixU() * singular_values.asDiagonal() * svd.matrixV().transpose();
    
    // F = T1.transpose()*F*T0;
    // PRINT("F", F);

    // cv::Mat F_cv = cv::findFundamentalMat(p0_cv, p1_cv);
    // PRINT(F_cv);
    // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    //     F(F_cv.ptr<double>(), F_cv.rows, F_cv.cols);

    // // PRINT(T1.transpose()*F*T0);  
    // // PRINT("adjkw");
    // Eigen::MatrixXd l = (F.transpose() * p0_sel_homo.transpose()).transpose();
    // // Eigen::MatrixXd l_prime = (F.transpose() * p1_sel_homo.transpose()).transpose();

    // double sx = -2000;
    // double ex = 2000;
    //  // ax+by+c = 0, y = -(a*x + c)/b
    // for(int i=0; i<8; ++i){
    //     double a = l(i,0);
    //     double b = l(i,1);
    //     double c = l(i,2);

    //     // double a_prime = l_prime(i,0);
    //     // double b_prime = l_prime(i,1);
    //     // double c_prime = l_prime(i,2);

    //     double sp = -1*(a*sx + c)/b;
    //     double ep = -1*(a*ex + c)/b;
    //     // PRINT(sp, ep);
    //     cv::line(imgs[1], cv::Point2d(sx, sp), cv::Point2d(ex, ep), 
    //                 cv::Scalar(255,0,0), 1, cv::LINE_AA);

    //     // sp = -1*(a_prime*sx + c_prime)/b_prime;
    //     // ep = -1*(a_prime*ex + c_prime)/b_prime;
    //     // // PRINT(sp, ep);
    //     // cv::line(imgs[0], cv::Point2d(sx, sp), cv::Point2d(ex, ep), 
    //     //             cv::Scalar(0,0,255), 1, cv::LINE_AA);
    // }


    // cv::drawKeypoints(imgs[0], kpts0, imgs[0]);
    // cv::drawKeypoints(imgs[1], kpts1, imgs[1]);
    // cv::Mat merged;
    // cv::hconcat(imgs[0], imgs[1], merged);
    // cv::imshow("img", merged);
    // cv::waitKey();







    return 1;
}