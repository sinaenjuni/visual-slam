# include "intrinsic_parameter.h"

void find_image_point(cv::Mat &img, cv::Mat &corners, const int col, const int row){
    cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
    cv::findChessboardCorners(img, cv::Size(col, row), corners);
    cv::TermCriteria criteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 30, 0.001);
    cv::cornerSubPix(img, corners, cv::Size(3,3), cv::Size(-1, -1), criteria);
}


cv::Mat compute_H(cv::Mat &src_points, cv::Mat &dst_points, const int col, const int row){
    cv::Mat A(2*row*col, 9, CV_32F);
    for(int i=0; i<row*col; ++i){
        const cv::Point2f src = src_points.at<cv::Point2f>(i);  // obj point
        const cv::Point2f dst = dst_points.at<cv::Point2f>(i);  // img point
        // PRINT(src);
        // PRINT(dst);
        double X = src.x; double Y = src.y;
        double u = dst.x; double v = dst.y;
        A.at<cv::Vec<double, 9>>(i*2) << -X, -Y, -1, 0, 0, 0, u*X, u*Y, u;
        A.at<cv::Vec<double, 9>>(i*2+1) << 0, 0, 0, -X, -Y, -1, v*X, v*Y, v;
    }

    cv::Mat w, u, vt;
    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    vt = vt.row(8).reshape(0, 3);
    vt = vt / vt.at<float_t>(2,2);
    return vt;
}

Eigen::Matrix3d ecompute_H(cv::Mat &src_points, cv::Mat &dst_points, const int col, const int row)
{
    // Eigen::Matrix<double, 2*row*col, 9> A = Eigen::Matrix<double, 2*row*col, 9>::Constant(0);
    Eigen::MatrixXd A(2 * row * col, 9);
    A.setConstant(0);
    for(int i=0; i<row*col; ++i){
        const cv::Point2f src = src_points.at<cv::Point2f>(i);  // obj point
        const cv::Point2f dst = dst_points.at<cv::Point2f>(i);  // img point
        const double X = src.x; double Y = src.y;
        const double u = dst.x; double v = dst.y;
        A.block(i*2, 0, 1, 9) << -X, -Y, -1, 0, 0, 0, u*X, u*Y, u;
        A.block(i*2+1, 0, 1, 9) << 0, 0, 0, -X, -Y, -1, v*X, v*Y, v;
    }
    const Eigen::JacobiSVD svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> H(svd.matrixV().col(8).data());
    H = H / H(2, 2);
    return H;
}

void normalize(const cv::Mat &points, cv::Mat &T, const int col, const int row)
{
    double x_mean=0, y_mean=0;
    for(int i=0; i<row*col; ++i){
        const cv::Point2f p = points.at<cv::Point2f>(i);
        x_mean += p.x;
        y_mean += p.y;
    }
    x_mean /= row*col;
    y_mean /= row*col;
    
    double x_mean_dist = 0;
    double y_mean_dist = 0;
    for(int i=0; i<row*col; ++i){
        const cv::Point2f p = points.at<cv::Point2f>(i);
        x_mean_dist += fabs(p.x - x_mean);
        y_mean_dist += fabs(p.y - y_mean);
    }
    x_mean_dist /= row*col;
    y_mean_dist /= row*col;

    double xs = 1.0/x_mean_dist;
    double ys = 1.0/y_mean_dist;

    T = (cv::Mat_<double>(3,3) << xs, 0, -xs*x_mean, 
                                    0, ys, -ys*y_mean,
                                    0, 0, 1);
}
