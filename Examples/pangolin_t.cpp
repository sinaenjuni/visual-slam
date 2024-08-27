#include <iostream>
#include <unistd.h>

#include "pangolin/pangolin.h"
#include "Eigen/eigen"
#include "sophus/se3.hpp"

#include "file.h"
#include "thread"
#include <chrono>

void increase(int &move){
    while (1)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ++move;

    }
}


void render_view(std::string title){
    pangolin::CreateWindowAndBind(title,1024,768);
    glEnable(GL_DEPTH_TEST);

    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    std::vector<int> degrees{1, 2 , 4, 8};
    
    Eigen::Matrix3d R;
    double theta = M_PI / 2; // 45도
    R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY());

    // 카메라 위치와 방향 설정
    Eigen::Vector3d eye(0, 0, 1);  // 카메라의 초기 위치 (z 방향 1)
    Eigen::Vector3d center = R * Eigen::Vector3d(0, 0, -1);  // 카메라가 바라보는 방향
    Eigen::Vector3d up = R * Eigen::Vector3d(0, 1, 0);  // 상향 
    
    // Eigen::Matrix4d m;
    // m << 0,10,0.1, center[0], center[1], center[2], 0.0,-1.0, 0.0;

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024,768, 2000,2000, 512,389,0.1,1000),
        pangolin::ModelViewLookAt(0, 20, 0.1, 0,0,1 , 0.0,2.0, 0.0)
    );


    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            // .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(&handler);

    Eigen::Matrix3d K;
    K = K.Identity();
    // K(0,0) = 1;
    // K(1,1) = 1;
    K(0,2) = 1;
    K(1,2) = 1;
    PRINT(K);
    
    Eigen::Matrix3d T_r = Eigen::AngleAxis(M_PI/4, Eigen::Vector3d(0, 1, 0)).toRotationMatrix();
    Sophus::SE3d T;
    T.setRotationMatrix(T_r);

    // PRINT(T.rotationMatrix());
    // PRINT(T.translation());

    // PRINT(T);


    Eigen::Matrix4d eigen_matrix = Eigen::Map<Eigen::Matrix4d>(s_cam.GetModelViewMatrix().m);

    int move = 0;
    std::thread t(increase, std::ref(move));


    while(!pangolin::ShouldQuit()){

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        
        pangolin::glDrawAxis(1);
    
        // T.translation() = Eigen::Vector3d(1.0, 0, 0);
        // pangolin::glDrawFrustum<double>(K.inverse().matrix(), 2, 2, T.matrix(), 1);
        
        T.translation() = Eigen::Vector3d(0, 0, move);
        pangolin::glDrawFrustum<double>(K.inverse().matrix(), 2, 2, T.matrix(), 1);

        // Sophus::SE3d new_cam = T; 
        // new_cam.translation().matrix() - Eigen::Vector3d( 0, 0, 0.5);

        pangolin::OpenGlMatrix cam_model = s_cam.GetModelViewMatrix();
        Eigen::Matrix4d cam_model_eigen = Eigen::Map<Eigen::Matrix4d>(cam_model.m);
        // cam_model_eigen.block(0,0,3,3) *= T.rotationMatrix().matrix();
        // PRINT(cam_model_eigen.block(0,3, 3, 1));
        // PRINT(T.translation().matrix());
        // cam_model_eigen.block(0,3, 3, 1) = T.translation().matrix();
        // eigen_matrix = eigen_matrix * T.matrix().inverse();

        // Eigen::Matrix3d R_cam = Eigen::AngleAxis(M_PI * 0.2, Eigen::Vector3d(0, 1, 0)).toRotationMatrix() * T.rotationMatrix(); 
        Eigen::Matrix3d R_cam = T.rotationMatrix(); 
        cam_model_eigen.block(0,0, 3, 3) = R_cam;
        PRINT(s_cam.GetModelViewMatrix());
        PRINT(T.rotationMatrix());
        PRINT(T.translation());

        pangolin::glDrawAxis(T.matrix(), 1);
        // PRINT(eigen_matrix);
        s_cam.SetModelViewMatrix(cam_model_eigen);
        s_cam.Follow(T.matrix());
        // Eigen::Matrix4d mm;
        // pangolin::OpenGlMatrix m();
        // m.SetIdentity();

        // s_cam.SetModelViewMatrix(m);
        // for(int d : degrees){
        // s_cam.SetModelViewMatrix(m);
        // }

        // s_cam.

        // pangolin::glDrawColouredCube();
    
        pangolin::FinishFrame();
    }
    t.detach();

}   

int main() {
    // auto resultFuture = std::async(std::launch::async, render_view, "Main");
    // resultFuture.get();

    render_view("main");


    
    return 0;
}