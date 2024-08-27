#include "Eigen/eigen"
#include "Opencv2/opencv.hpp"


void find_image_point(cv::Mat &img, cv::Mat &corners, const int col, const int row);
cv::Mat compute_H(cv::Mat &src_points, cv::Mat &dst_points, const int col, const int row);
Eigen::Matrix3d ecompute_H(cv::Mat &src_points, cv::Mat &dst_points, const int col, const int row);
void normalize(const cv::Mat &points, cv::Mat &T, const int col, const int row);
