cmake_minimum_required(VERSION 3.27)
project(project4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(project4
        src/main.cpp
        src/target_detection.cpp
        src/calibration_utils.cpp
        src/camera_calibration.cpp
)

add_executable(project4_pose
        src/pose_main.cpp
        src/target_detection.cpp
        src/calibration_utils.cpp
        src/pose_estimation.cpp
)

target_include_directories(project4 PRIVATE include)
target_include_directories(project4_pose PRIVATE include)

target_link_libraries(project4 PRIVATE ${OpenCV_LIBS})
target_link_libraries(project4_pose PRIVATE ${OpenCV_LIBS})