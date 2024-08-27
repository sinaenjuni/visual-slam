#include "pangolin/pangolin.h"
#include "frame.h"

class Viewer
{
private:
    int WIDTH, HEIGHT;
    Eigen::Matrix3d KinvE;
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
public:
    Viewer(int WIDTH, int HEIGHT, Eigen::Matrix3d KinvE);
    void draw(std::vector<Frame<cv::ORB>> frames);
    ~Viewer();
};