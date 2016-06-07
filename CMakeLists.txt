cmake_minimum_required(VERSION 3.5)
project(DeblurringCV)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp cvTools.h cvDeconv.h shift.hpp cvQM.h)
add_executable(DeblurringCV ${SOURCE_FILES})
target_link_libraries(DeblurringCV opencv_core opencv_highgui opencv_flann opencv_imgproc)