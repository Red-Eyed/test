#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#define MAX_SIZE 10024

using namespace cv;
int main()
{
    ssize_t num = 8;
    Ptr<cuda::TemplateMatching> matcher = cv::cuda::createTemplateMatching(CV_8U, cv::TemplateMatchModes::TM_CCOEFF_NORMED);

    std::vector<cuda::Stream> streams(num);
    cv::Mat img(MAX_SIZE, MAX_SIZE, CV_8U);
    cv::Mat img2(MAX_SIZE / 10, MAX_SIZE / 10, CV_8U);
    cuda::GpuMat gpuImage, templ, gpuImage3;
    gpuImage.upload(img);
    templ.upload(img2);
    gpuImage3.upload(img);

    for (auto&& stream : streams) {
        matcher->match(gpuImage, templ, gpuImage3);
    }
    for (auto&& stream : streams) {
        stream.waitForCompletion();
    }
    std::cout << cv::getBuildInformation() << std::endl;
}

//int main()
//{
//    UMat img(MAX_SIZE, MAX_SIZE, CV_32FC3);

//    for (int j = 0; j < 1000; j++) {
//        add(img, img, img);
//    }
//}
