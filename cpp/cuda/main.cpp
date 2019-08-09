#include <iostream>
#include <omp.h>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>
#include <unistd.h>
#include <vector>

#define MAX_SIZE 1000

using namespace cv;
int main()
{
    int num = 10;

    std::vector<cuda::Stream> streams;
    for (int i = 0; i < num; ++i) {
        streams.push_back(cuda::Stream());
    }

#pragma omp parallel num_threads(10)
    {
        int i = omp_get_thread_num();
        cuda::GpuMat gpuImage(MAX_SIZE, MAX_SIZE, CV_32FC3);
        for (int j = 0; j < 1000; j++) {
            cuda::add(gpuImage, gpuImage, gpuImage, noArray(), -1, streams[i]);
        }
        streams[i].waitForCompletion();
    }
}
