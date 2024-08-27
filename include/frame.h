#ifndef FRAME_H
#define FRAME_H

#include "Eigen/eigen"
#include "sophus/se3.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core/eigen.hpp>

#include "file.h"

// Eigen::Matrix4d triangulate(Eigen::Matrix4d pose0, Eigen::Matrix4d pose1, 
//                             cv::Mat points0, cv::Mat points1){
//     Eigen::Matrix A(points0)
//     ret = np.zeros((pts1.shape[0], 4))
//     for i, p in enumerate(zip(pts1, pts2)):
//     A = np.zeros((4,4))
//     A[0] = p[0][0] * pose1[2] - pose1[0]
//     A[1] = p[0][1] * pose1[2] - pose1[1]
//     A[2] = p[1][0] * pose2[2] - pose2[0]
//     A[3] = p[1][1] * pose2[2] - pose2[1]
//     _, _, vt = np.linalg.svd(A)
//     ret[i] = vt[3]
//     return ret
// }

template <typename T>
class Frame
{
private:
    cv::Mat image;
    cv::Mat image_points, obj_points;
    cv::Ptr<T> feature_extractor;
    cv::BFMatcher matcher;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<Eigen::Vector4d> obj_pointsE;

    cv::Mat K, R, t, Tcw; 
    Eigen::MatrixXd eigenK;
    Sophus::SE3d Twc;


public:
    Frame(cv::Mat image, cv::Ptr<T> feature_extractor, cv::BFMatcher matcher, cv::Mat K);
    std::vector<cv::KeyPoint> getKeyPoints();
    Sophus::SE3d getTwc();
    std::vector<Eigen::Vector4d> getObjPoints();
    void calcPoseWithF(Frame target_frame);
    ~Frame();
};

template <typename T>
Frame<T>::Frame(cv::Mat image, cv::Ptr<T> feature_extractor, cv::BFMatcher matcher, cv::Mat K) : 
                image(image), feature_extractor(feature_extractor), matcher(matcher), K(K)
{
    R = cv::Mat::eye(cv::Size(3,3), CV_64F);
    t = cv::Mat::zeros(cv::Size(1,3), CV_64F);
    cv::hconcat(R, t, Tcw);

    cv::cv2eigen(K, eigenK);
    Twc = Sophus::SE3d();

    cv::Mat image_gray;
    if(image.channels() != 1){
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    }else{
        image.copyTo(image_gray);
    }
    feature_extractor->detectAndCompute
        (image_gray, cv::noArray(), keypoints, descriptors);
}

template <typename T>
inline std::vector<cv::KeyPoint> Frame<T>::getKeyPoints()
{
    return keypoints;
}

template <typename T>
Sophus::SE3d Frame<T>::getTwc(){
    return Twc;
}

template <typename T>
inline void Frame<T>::calcPoseWithF(Frame target_frame)
{
    std::vector<cv::DMatch> matches;
    matcher.match(this->descriptors, target_frame.descriptors, matches);
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
        return a.distance < b.distance;
    });

    std::vector<cv::Point2d> this_matched_points, target_matched_points;
    for(int i=0; i<matches.size(); ++i){
        cv::Point2d p0 = target_frame.keypoints[matches[i].trainIdx].pt;
        cv::Point2d p1 = this->keypoints[matches[i].queryIdx].pt;
        if(cv::norm(p0-p1) > 50){
            continue;
        }
        target_matched_points.push_back(p0);
        this_matched_points.push_back(p1);
    }
    
    cv::Mat F, inlier;
    F = cv::findFundamentalMat(target_matched_points, this_matched_points, inlier, cv::FM_RANSAC, 3., 0.99);    
    cv::Mat E;
    E = K.t() * F * K;
    // E = cv::findEssentialMat(target_matched_points, this_matched_points, K, cv::RANSAC, 0.999, 1.0, 1000, inlier);

    cv::recoverPose(E, target_matched_points, this_matched_points, K, R, t, inlier);
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    this->Tcw = K*Rt;

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    eigenR(R.ptr<double>(), R.rows, R.cols);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    eigent(t.ptr<double>(), t.rows, t.cols);

    this->Twc = target_frame.getTwc() * Sophus::SE3d(eigenR, eigent).inverse();

    Eigen::Matrix4d pose0 = target_frame.eigenK * target_frame.Twc.inverse().matrix();
    Eigen::Matrix4d pose1 = this->eigenK * this->Twc.inverse().matrix();

    for(int i=0; i<this_matched_points.size(); ++i)
    {
        Eigen::Matrix4d A;
        cv::Point2d target_point = target_matched_points[i];
        cv::Point2d this_point = this_matched_points[i];

        Eigen::Vector2d p0(target_point.x, target_point.y);
        Eigen::Vector2d p1(this_point.x, this_point.y);

        A.row(0) = p0(0) * pose0.row(2) - pose0.row(0);
        A.row(1) = p0(1) * pose0.row(2) - pose0.row(1);
        A.row(2) = p1(0) * pose1.row(2) - pose1.row(0);
        A.row(3) = p1(1) * pose1.row(2) - pose1.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4d X = svd.matrixV().col(3);

        X /= X(3);
        if(X[2] < 0) continue;
        this->obj_pointsE.push_back(X);
    }


    // PRINT(this->P);
    // PRINT(target_frame.P);
    // cv::triangulatePoints(this->P, target_frame.P, this_matched_points, target_matched_points, this->obj_points);
    // cv::triangulatePoints(target_frame.P, this->P, target_matched_points, this_matched_points, this->obj_points);
    // this->obj_points.row(0) = this->obj_points.row(0) / this->obj_points.row(3);
    // this->obj_points.row(1) = this->obj_points.row(1) / this->obj_points.row(3);
    // this->obj_points.row(2) = this->obj_points.row(2) / this->obj_points.row(3);
    // this->obj_points.row(3) = 1;

    // cv::convertPointsFromHomogeneous(this->obj_points.t(), this->obj_points);
    // CVSH(this->obj_points);
    // PRINT(this->obj_points.row(0));
    // PRINT(this->obj_points.colRange(cv::Range(0,this->obj_points.cols))/this->obj_points.row(3));
    // CVSH(this->obj_points);
    // this->obj_points = this->obj_points.t();
    // CVSH(this->obj_points);

    // this->obj_points = this->obj_points.t();
    // for(int i=0; i<this->obj_points.rows; ++i){
    //     cv::Vec4d point = this->obj_points.row(i);
    //     point = point/point[3];

    //     // if(point[2] <= 0 || point[2] > 20)  continue;
    //     // if(point[2] <= 0)  continue;
    //     // PRINT(point);


    //     Eigen::Vector4d pointE;
    //     cv::cv2eigen(point, pointE);
    //     this->obj_pointsE.push_back(pointE);
    // }

    // for(cv::Vec4d obj_point : this->obj_points){
        // PRINT(obj_point);
        // this->obj_pointsE.push_back(obj_points);
    // }


    // cv::imshow("this", this->image);
    // cv::imshow("target", target_frame.image);

    cv::Mat merge;
    cv::vconcat(target_frame.image, this->image, merge);
    cv::imshow("merge", merge);
}

template <typename T>
inline std::vector<Eigen::Vector4d> Frame<T>::getObjPoints()
{
    return obj_pointsE;
}

template <typename T>
Frame<T>::~Frame()
{
}


#endif // FRAME_H