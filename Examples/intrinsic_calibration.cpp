#include <iostream>
#include <vector>
#include <format>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "Eigen/eigen"
#include <opencv2/core/eigen.hpp>

#include "file.h"
#include "intrinsic_parameter.h"

constexpr int ROW = 9, COL=7, SQUARE_SIZE=20;
constexpr int HEIGHT = 901, WIDTH = 1600;


int main(int, char**){
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
    const int N = imgs.size();

    // make world points
    cv::Mat world_point(ROW*COL, 1, CV_32FC2);
    for(int i=0; i<ROW*COL; ++i){
        const float_t row = i % COL;
        const float_t col = i / COL;
        world_point.at<cv::Vec<float_t, 2>>(i) << row*SQUARE_SIZE, col*SQUARE_SIZE;
    }

    // compute Homography
    std::vector<Eigen::Matrix3d> Hs;
    for(int i=0; i<imgs.size(); ++i){
        cv::Mat image_point; // CV_32FC2, 64 x 1 x 2
        find_image_point(imgs[i], image_point, COL, ROW);

        // cv::Mat Hiw = compute_H(world_point, image_point);
        const Eigen::Matrix3d Hiw = ecompute_H(world_point, image_point, COL, ROW);
        Hs.push_back(Hiw);
    }
    // for(const auto &H: Hs)
    // {
    //     PRINT(H);
    // }
    
    // // cv::Mat A(Hs.size()*2, 6, CV_32F);
    Eigen::MatrixXd A(2*N, 6);
    A.setConstant(0);
    for(int i=0; i<imgs.size(); ++i){
        const DTYPE h1 = Hs[i](0, 0);
        const DTYPE h2 = Hs[i](0, 1);
        const DTYPE h3 = Hs[i](0, 2);
        const DTYPE h4 = Hs[i](1, 0);
        const DTYPE h5 = Hs[i](1, 1);
        const DTYPE h6 = Hs[i](1, 2);
        const DTYPE h7 = Hs[i](2, 0);
        const DTYPE h8 = Hs[i](2, 1);
        const DTYPE h9 = Hs[i](2, 2);
        A.block(i*2, 0, 1, 6) << h1*h2, h1*h5 + h2*h4, h1*h8 + h2*h7, h4*h5, h4*h8 + h5*h7, h7*h8;
        A.block(i*2+1, 0, 1, 6) << h1*h1-h2*h2,2*h1*h4-2*h2*h5,2*h1*h7-2*h2*h8,h4*h4-h5*h5,2*h4*h7-2*h5*h8,h7*h7-h8*h8;
    }
    // PRINT(A);
    
    Eigen::JacobiSVD svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector<double, 6> a(svd.matrixV().col(5).data());
    
    Eigen::Matrix3d KtinvKinv;
    KtinvKinv << a(0), a(1), a(2), a(1), a(3), a(4), a(2), a(4), a(5);
    PRINT(KtinvKinv);

    Eigen::Matrix3d pseudoKinv = KtinvKinv.llt().matrixL();
    Eigen::Matrix3d K = pseudoKinv.inverse().transpose();
    K = K / K(2, 2);
    Eigen::Matrix3d Kinv = K.inverse();
    std::ofstream file("../data/output.txt");
    if (!file.is_open()) {
        std::cerr << "Unable to open file for writing: " << std::endl;
    }
    file <<"K:\n";
    file << K << "\n";
    
    std::vector<Eigen::Matrix3d> Rs(imgs.size());
    std::vector<Eigen::Vector3d> ts(imgs.size());
    PRINT(K);
    PRINT(Kinv);

    PRINT("--------------------------------------");
    for (int i = 0; i < imgs.size(); i++) {
        Eigen::Vector3d r1 = Kinv * Hs[i].col(0);
        Eigen::Vector3d r2 = Kinv * Hs[i].col(1);
        Eigen::Vector3d r3 = r1.cross(r2);
        Eigen::Matrix3d R;
        R << r1, r2, r3;

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();

        Rs[i] = R;

        double lambda = 1 / r1.norm();
        Eigen::Vector3d t = lambda * Kinv * Hs[i].col(2);

        ts[i] = t;

        Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
        P.block(0,0,3,3) = R;
        P.block(0,3,3,1) = t;
        PRINT(P);

        PRINT("i: " << i );
        PRINT("R\n" << R );
        PRINT("t\n" << t );
        PRINT("--------------------------------------" );
        file << "IMG: " << i << "\n";
        file << P << "\n";
        // file <<"R:\n";
        // file << R << "\n";
        // file <<"t:\n";
        // file << t << "\n";

    }
    file.close();

    return 0;
}

// void saveToTextFile(const std::vector<Eigen::Matrix3d>& matrices, const std::string& filename) {
//     std::ofstream file(filename);

//     if (file.is_open()) {
//         for (const auto& mat : matrices) {
//             file << mat << "\n\n";  // 행렬을 파일에 쓰기
//         }
//         file.close();
//     } else {
//         std::cerr << "Unable to open file for writing: " << filename << std::endl;
//     }
// }

