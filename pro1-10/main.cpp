#include <iostream>
#include <pro1-10.hpp>

int main(int argc, char const *argv[]) {
  std::cout << "hello world \n";
  cv::Mat img = cv::imread("../../data/imori.jpg");
  cv::Mat rgb = bgr2rgb(img);
  cv::imwrite("../data/answer1.jpg", rgb);
  cv::Mat gray = bgr2gray(img);
  cv::imwrite("../data/answer2.jpg", gray);
  cv::Mat binary = thresholding(img, 128);
  cv::imwrite("../data/answer3.jpg", binary);
  int threshold;
  cv::Mat ostu_img = otsu_thresholding(img, threshold);
  cv::imwrite("../data/answer4.jpg", ostu_img);
  cv::Mat averpool_img = average_pooling(img, 8);
  cv::imwrite("../data/answer5.jpg", averpool_img);
  cv::Mat maxpool_img = max_pooling(img, 8);
  cv::imwrite("../data/answer6.jpg", maxpool_img);

  cv::Mat noise_img = cv::imread("../../data/imori_noise.jpg");
  cv::Mat gauss_filter_img = gaussian_filter(noise_img, 3, 0.8);
  cv::imwrite("../../data/answer7.jpg", gauss_filter_img);
  cv::Mat median_filter_img = median_filter(noise_img, 3);
  cv::imwrite("../../data/answer8.jpg", median_filter_img);
  return 0;
}
