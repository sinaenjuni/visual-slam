#include "viewer.h"

Viewer::Viewer(int WIDTH, int HEIGHT, Eigen::Matrix3d KinvE) : 
    WIDTH(WIDTH), HEIGHT(HEIGHT), KinvE(KinvE)
{
    pangolin::CreateWindowAndBind("Main",1024,768);
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024,768, 2000,2000, 512,389,0.1,1000),
        pangolin::ModelViewLookAt(0,-100,-0.1, 0,0,0, 0.0,-1.0, 0.0)
        // pangolin::ModelViewLookAt(0,0.01,50, 0,0,0,0.0,0.0, 1.0)
    );

    // pangolin::Handler3D handler(s_cam);
    d_cam = pangolin::CreateDisplay()
            // .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f/768.0f)
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            // .SetHandler(&handler);
            .SetHandler(new pangolin::Handler3D(s_cam));


    // pangolin::View& d_image = pangolin::Display("image")
    // .SetBounds(2/3.0f,1.0f,0,1/3.0f,-static_cast<float>(WIDTH)/static_cast<float>(HEIGHT))
    // .SetLock(pangolin::LockLeft, pangolin::LockTop);

    // unsigned char* imageArray = new unsigned char[3*WIDTH*HEIGHT];
    // pangolin::GlTexture imageTexture(WIDTH,HEIGHT,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
}

void Viewer::draw(std::vector<Frame<cv::ORB>> frames)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);

    pangolin::glDrawAxis(1);

    for(int i=0; i<frames.size(); ++i){
        Frame frame = frames[i];
        // pangolin::glDrawFrustum<double>(KinvE.matrix(), WIDTH, HEIGHT, Twcs[i].matrix(), 1);
        glColor4f(1.0, 1.0, 1.0, 1.0);
        pangolin::glDrawFrustum<double>(KinvE.matrix(), WIDTH, HEIGHT, frame.getTwc().matrix(), 1);
        // pangolin::glDrawAxis(Twcs[i].matrix(), 1);

        // glColor4f(1.0, 1.0, 1.0, 1.0);
        pangolin::glDrawAxis(frame.getTwc().matrix(), 1);

        glPointSize(1);
        glColor4f(0.0, 1.0, 0.0, 0.6);
        pangolin::glDrawPoints(frame.getObjPoints());
    }

    // glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    // imageTexture.Upload(frame_curr.data, GL_RGB, GL_UNSIGNED_BYTE);

    // d_image.Activate();
    // imageTexture.RenderToViewportFlipY();

    if(frames.size() != 0){
        s_cam.Follow(frames.back().getTwc().matrix());    
    }
    pangolin::FinishFrame();
}

Viewer::~Viewer()
{
}