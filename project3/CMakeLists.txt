cmake_minimum_required(VERSION 3.16)
project(project3 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# common helper library (segmentation, features, db I/O, classifier, visualization, pipeline)
add_library(p3_common STATIC
        src/p3_segmentation.cpp
        src/p3_db.cpp
        src/p3_classifier.cpp
        src/p3_visualize.cpp
        src/p3_pipeline.cpp)

target_include_directories(p3_common PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS})

target_link_libraries(p3_common PUBLIC ${OpenCV_LIBS})

# --------  p3_main --------
add_executable(main
        src/main.cpp)

target_include_directories(main PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(main PRIVATE ${OpenCV_LIBS} p3_common)