cmake_minimum_required(VERSION 3.5.0)
project(visual_slam VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

include_directories(include)
add_library(${PROJECT_NAME} SHARED 
src/file.cpp
include/file.h
src/intrinsic_parameter.cpp
include/intrinsic_parameter.h
src/viewer.cpp
include/viewer.h
src/frame.cpp
include/frame.h
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# add_executable(visual_slam main.cpp)
# add_executable(main Examples/main.cpp)
# add_executable(homography Examples/homography.cpp)
# add_executable(intrinsic_calibration Examples/intrinsic_calibration.cpp)

find_package(OpenCV REQUIRED PATHS "Thirdparty/OpenCV")
if(OpenCV_FOUND)
    message("OpenCV ${OpenCV_VERSION} found")
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    # include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message("OpenCV not found")
endif()

find_package(Eigen3 REQUIRED PATHS "Thirdparty/Eigen")
if(Eigen3_FOUND)
    message("Eigen3 ${Eigen3_VERSION} found")
    message(${EIGEN3_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
    include_directories(Eigen3::Eigen)
    # same functions
    # target_link_directories(homography PRIVATE Eigen3::Eigen)
    # include_directories(${EIGEN3_INCLUDE_DIRS}) 
else()
    message("Eigen3 not found")
endif()

find_package(Pangolin PATHS "Thirdpaty/Pangolin")
if(Pangolin_FOUND)
    message("Pangolin ${Pangolin_VERSION} found")
    target_link_libraries(${PROJECT_NAME} ${Pangolin_LIBRARY})
    include_directories(${Pangolin_INCLUDE_DIRS})
endif()

find_package(Sophus PATHS "Thirdpaty/Sophus")
if(Sophus_FOUND)
    message("Sophus ${Sophus_VERSION} found")
    target_link_libraries(${PROJECT_NAME} Sophus::Sophus)
    include_directories(Sophus::Sophus)
endif()

find_package(G2o PATHS "Thirdpaty/G2o")
if(G2o_FOUND)
    message("G2o ${G2o_VERSION} found")
    target_link_libraries(${PROJECT_NAME} g2o::core g2o::solver_dense)
    # target_link_libraries(${PROJECT_NAME} g2o::solver_dense)
    # include_directories(g2o::core)
endif()


# add_executable(visual_slam main.cpp)
add_executable(main Examples/main.cpp)
target_link_libraries(main ${PROJECT_NAME})

add_executable(homography Examples/homography.cpp)

add_executable(intrinsic_calibration Examples/intrinsic_calibration.cpp)
target_link_libraries(intrinsic_calibration ${PROJECT_NAME})

add_executable(pangolin_t Examples/pangolin_t.cpp)
target_link_libraries(pangolin_t ${PROJECT_NAME})

add_executable(hellopangolin Examples/hellopangolin.cpp)
target_link_libraries(hellopangolin ${PROJECT_NAME})

add_executable(circle_fit Examples/circle_fit.cpp)
target_link_libraries(circle_fit ${PROJECT_NAME})

add_executable(essential_triangulation Examples/essential_triangulation.cpp)
target_link_libraries(essential_triangulation ${PROJECT_NAME})
