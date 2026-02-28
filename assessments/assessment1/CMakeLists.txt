cmake_minimum_required(VERSION 3.5)
project(EntropyDemo)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(entropy_demo src/entropy_demo.cpp)
target_link_libraries(entropy_demo ${OpenCV_LIBS})

add_executable(patch_entropy src/patch_entropy.cpp)
target_link_libraries(patch_entropy ${OpenCV_LIBS})

add_executable(entropy_map src/entropy_map.cpp)
target_link_libraries(entropy_map ${OpenCV_LIBS})