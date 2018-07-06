#include "opencv2/opencv.hpp"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"

void OpencvMat2ArmCLImage(const cv::Mat &mat, arm_compute::Image &image)
{
  if(mat.cols != image.info()->dimension(0) ||
      mat.rows != image.info()->dimension(1))
  {
    printf("cv::Mat size %dx%d and arm_compute::Image %dx%d not match!\n", mat.cols, mat.rows, image.info()->dimension(0), image.info()->dimension(1));
    return;
  }

  unsigned char* mat_ptr;
  arm_compute::Window window;
  arm_compute::Iterator out(&image, window);

  window.set(arm_compute::Window::DimX, arm_compute::Window::Dimension(0, mat.cols, 1));
  window.set(arm_compute::Window::DimY, arm_compute::Window::Dimension(0, mat.rows, 1));

  mat_ptr = mat.data;
  arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
  {
    *out.ptr() = *mat_ptr++;
  },
  out);
}
