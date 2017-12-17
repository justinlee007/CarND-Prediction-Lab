cmake_minimum_required(VERSION 3.7)
project(CarND_Prediction_Lab)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/main.cpp src/classifier.cpp src/classifier.h)

include_directories(src/Eigen-3.3)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(CarND_Prediction_Lab ${SOURCE_FILES})