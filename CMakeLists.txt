cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 17)

project(matchcdf)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) # provide library headers
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}) # link OpenCV libraries
