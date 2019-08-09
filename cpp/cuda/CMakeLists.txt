cmake_minimum_required(VERSION 3.0)

get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
string(REPLACE " " "_" PROJECT_NAME ${PROJECT_NAME})
project(${PROJECT_NAME} CXX)
add_definitions("-DCUDA_API_PER_THREAD_DEFAULT_STREAM")

set(CMAKE_VERBOSE_MAKEFILE ON)
add_executable(${PROJECT_NAME} "main.cpp")
find_package( OpenCV REQUIRED )
find_package( OpenMP REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( ${PROJECT_NAME} PUBLIC ${OpenCV_LIBS}  OpenMP::OpenMP_CXX )
