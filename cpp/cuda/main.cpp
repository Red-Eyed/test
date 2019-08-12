#include <opencv4/opencv2/opencv.hpp>
#include <vector>

#define MAX_SIZE 1024

using namespace cv;
int main()
{
    ssize_t num = 8;

    std::vector<cuda::Stream> streams(num);
    cuda::GpuMat gpuImage(MAX_SIZE, MAX_SIZE, CV_32FC3);

    for (auto&& stream : streams) {
        for (int j = 0; j < 10; j++) {
            cuda::add(gpuImage, gpuImage, gpuImage, noArray(), -1, stream);
        }
    }
    for (auto&& stream : streams) {
        stream.waitForCompletion();
    }
}
