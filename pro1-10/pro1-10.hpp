#include <opencv2/opencv.hpp>

cv::Mat bgr2rgb(const cv::Mat &bgr) {
  assert(bgr.data);
  assert(bgr.channels() == 3);
  cv::Mat out(bgr.size(), bgr.type());
  int chan = bgr.channels();
  int height = bgr.rows;
  int width = bgr.cols;
  for (int ih = 0; ih < height; ih++) {
    auto sptr = bgr.ptr<uchar>(ih);
    auto dptr = out.ptr<uchar>(ih);
    for (size_t iw = 0; iw < width; iw++) {
      uchar b = sptr[chan * iw];
      uchar g = sptr[chan * iw + 1];
      uchar r = sptr[chan * iw + 2];
      dptr[chan * iw] = r;
      dptr[chan * iw + 1] = g;
      dptr[chan * iw + 2] = b;
    }
  }
  return out;
}

cv::Mat bgr2gray(const cv::Mat &image) {
  assert(image.data);
  assert(image.channels() == 3);
  int w = image.cols;
  int h = image.rows;
  int chan = image.channels();
  cv::Mat out(image.size(), CV_8UC1);
  for (int ih = 0; ih < h; ih++) {
    auto sptr = image.ptr<uchar>(ih);
    auto dptr = out.ptr<uchar>(ih);
    for (size_t iw = 0; iw < w; iw++) {
      uchar b = sptr[chan * iw];
      uchar g = sptr[chan * iw + 1];
      uchar r = sptr[chan * iw + 2];
      dptr[iw] = 0.21265 * r + 0.7125 * g + 0.0722 * b;
      // info: other way to get rgb
      //   out.at<uchar>(ih, iw) = 0.0722 * image.at<cv::Vec3b>(ih, iw)[0] +
      //                           0.7125 * image.at<cv::Vec3b>(ih, iw)[1] +
      //                           0.21265 * image.at<cv::Vec3b>(ih, iw)[2];
    }
  }
  return out;
}

cv::Mat thresholding(const cv::Mat &image, int threshold) {
  cv::Mat gray;
  if (image.channels() == 3) {
    gray = bgr2gray(image);
  } else {
    gray = image.clone();
  }
  cv::Mat out(gray.size(), gray.type());
  int h = image.rows;
  int w = image.cols;
  for (size_t i = 0; i < h; i++) {
    for (size_t j = 0; j < w; j++) {
      if (gray.at<uchar>(i, j) > threshold) {
        out.at<uchar>(i, j) = 255;
      } else {
        out.at<uchar>(i, j) = 0;
      }
    }
  }
  return out;
}

cv::Mat otsu_thresholding(const cv::Mat &image, int &threshhold) {
  cv::Mat gray;
  cv::Mat out(gray.size(), gray.type());
  int histogram[256] = {0};
  if (image.channels() == 3) {
    gray = bgr2gray(image);
  } else {
    gray = image.clone();
  }
  int h = image.rows;
  int w = image.cols;
  int image_size = h * w;
  double max_var = 0.0;
  double var;
  int count0;     // font back pix count(less than i)
  int count1;     // back gound pix count(more or equal than i)
  long sum0;      // font gound pix value sum(less than i)
  long sum1;      // bach gound pix value sum
  double u0, w0;  // font pix mean value and percentage
  double u1, w1;
  // static histogram
  for (size_t i = 0; i < h; i++) {
    auto sptr = gray.ptr<uchar>(i);
    for (size_t j = 0; j < w; j++) {
      histogram[sptr[j]]++;
    }
  }

  for (size_t i = 0; i < 256; i++) {
    sum0 = 0;
    sum1 = 0;
    count0 = 0;
    count1 = 0;
    u0 = 0.;
    w0 = 0.;
    u1 = 0.;
    w1 = 0.;
    var = 0.;
    // less than i
    for (size_t j = 0; j < i; j++) {
      count0 += histogram[j];
      sum0 += j * histogram[j];
    }
    w0 = (double)count0 / image_size;  // percentage of pix
    u0 = (double)sum0 / count0;        // mean  pix value of font
    // more or equal than i
    for (size_t j = i; j < 256; j++) {
      count1 += histogram[j];
      sum1 += j * histogram[j];
    }
    w1 = (double)count1 / image_size;  // percentage of pix
    u1 = (double)sum1 / count1;        // mean  pix value of font
    var = w0 * w1 * pow((u0 - u1), 2);
    if (var > max_var) {
      max_var = var;
      threshhold = i;
    }
  }
  std::cout << "threshold = " << threshhold << std::endl;
  return thresholding(image, threshhold);
}

cv::Mat reduce_color(const cv::Mat &image) {
  cv::Mat out(image.size(), image.type());
  for (size_t i = 0; i < image.rows; i++) {
    for (size_t j = 0; j < image.cols; j++) {
      out.at<cv::Vec3b>(i, j)[0] =
          image.at<cv::Vec3b>(i, j)[0] / 64 * 64 + 64 / 2;
      out.at<cv::Vec3b>(i, j)[1] =
          image.at<cv::Vec3b>(i, j)[1] / 64 * 64 + 64 / 2;
      out.at<cv::Vec3b>(i, j)[2] =
          image.at<cv::Vec3b>(i, j)[2] / 64 * 64 + 64 / 2;
    }
  }
  return out;
}

cv::Mat average_pooling(const cv::Mat &image, int block_size) {
  cv::Mat out(image.size(), image.type());
  for (int i = 0; i < image.rows; i += block_size) {
    for (int j = 0; j < image.cols; j += block_size) {
      for (int chan = 0; chan < image.channels(); chan++) {
        double sum = 0.0;
        for (int k = 0; k < block_size; k++) {
          for (int m = 0; m < block_size; m++) {
            sum += image.at<cv::Vec3b>(i + k, j + m)[chan];
          }
        }
        uchar mean = cv::saturate_cast<uchar>(sum / (block_size * block_size));

        for (int k = 0; k < block_size; k++) {
          for (int m = 0; m < block_size; m++) {
            out.at<cv::Vec3b>(i + k, j + m)[chan] = mean;
          }
        }
      }
    }
  }
  return out;
}

cv::Mat max_pooling(const cv::Mat &image, int block_size) {
  cv::Mat out(image.size(), image.type());
  for (int i = 0; i < image.rows / block_size; i++) {
    for (int j = 0; j < image.cols / block_size; j++) {
      for (int cha = 0; cha < image.channels(); cha++) {
        int max = 0;
        for (int m = 0; m < block_size; m++) {
          for (int n = 0; n < block_size; n++) {
            max = fmax(image.at<cv::Vec3b>(i * block_size + m,
                                           j * block_size + n)[cha],
                       max);
          }
        }

        for (int m = 0; m < block_size; m++) {
          for (int n = 0; n < block_size; n++) {
            out.at<cv::Vec3b>(i * block_size + m, j * block_size + n)[cha] =
                max;
          }
        }
      }
    }
  }
  return out;
}

cv::Mat create_gaussian_kernel(int kernel_size, double sigma) {
  cv::Mat kernel(kernel_size, kernel_size, CV_32FC1);
  int center = kernel_size / 2;
  float sum = 0.;
  int _x, _y;
  for (size_t i = 0; i < kernel_size; i++) {
    for (size_t j = 0; j < kernel_size; j++) {
      _x = j - center;
      _y = i - center;
      kernel.at<float>(i, j) = 1 / (2 * M_PI * sigma * sigma) *
                               exp(-(_x * _x + _y * _y) / (2 * sigma * sigma));

      sum += kernel.at<float>(i, j);
    }
  }
  kernel = kernel * (1 / sum);
  return kernel;
}

cv::Mat gaussian_filter(const cv::Mat &image, int kernel_size, double sigma) {
  // create kernel
  cv::Mat kernel = create_gaussian_kernel(kernel_size, sigma);
  std::cout << "kernel =" << kernel << std::endl;

  // zero padding
  int pad_size = kernel_size / 2;
  cv::Mat zero_pad_img(image.rows + pad_size * 2, image.cols + pad_size * 2,
                       image.type(), cv::Scalar(0, 0, 0));
  image.copyTo(
      zero_pad_img(cv::Rect(pad_size, pad_size, image.cols, image.rows)));

  cv::Mat out(zero_pad_img.size(), image.type());
  int center = kernel_size / 2;
  // filter
  for (size_t i = pad_size; i < zero_pad_img.rows - pad_size; i++) {
    for (size_t j = pad_size; j < zero_pad_img.cols - pad_size; j++) {
      for (size_t chan = 0; chan < zero_pad_img.channels(); chan++) {
        uchar value = 0;
        for (size_t m = 0; m < kernel_size; m++) {
          for (size_t n = 0; n < kernel_size; n++) {
            value += kernel.at<float>(m, n) *
                     zero_pad_img.at<cv::Vec3b>(i - center + m,
                                                j - center + n)[chan];
          }
        }
        out.at<cv::Vec3b>(i, j)[chan] = cv::saturate_cast<uchar>(value);
      }
    }
  }
  return out(cv::Rect(pad_size, pad_size, image.cols, image.rows)).clone();
}

cv::Mat median_filter(const cv::Mat &image, int kernel_size) {
  cv::imshow("org", image);
  cv::waitKey();
  // zero padding
  int pad_size = kernel_size / 2;
  cv::Mat zero_pad_img(image.rows + pad_size * 2, image.cols + pad_size * 2,
                       image.type(), cv::Scalar(0, 0, 0));
  image.copyTo(
      zero_pad_img(cv::Rect(pad_size, pad_size, image.cols, image.rows)));

  cv::Mat out(zero_pad_img.size(), image.type());
  int center = kernel_size / 2;
  for (size_t i = pad_size; i < zero_pad_img.rows - pad_size; i++) {
    for (size_t j = pad_size; j < zero_pad_img.cols - pad_size; j++) {
      for (size_t chan = 0; chan < zero_pad_img.channels(); chan++) {
        std::vector<uchar> arr;
        for (size_t m = 0; m < kernel_size; m++) {
          for (size_t n = 0; n < kernel_size; n++) {
            arr.push_back(zero_pad_img.at<cv::Vec3b>(i - center + m,
                                                     j - center + n)[chan]);
          }
        }
        std::sort(arr.begin(), arr.end());
        out.at<cv::Vec3b>(i, j)[chan] = arr[arr.size() / 2 + 1];
      }
    }
  }

  cv::imshow("median",
             out(cv::Rect(pad_size, pad_size, image.cols, image.rows)).clone());
  cv::waitKey();
  cv::destroyAllWindows();
  return out(cv::Rect(pad_size, pad_size, image.cols, image.rows)).clone();
}