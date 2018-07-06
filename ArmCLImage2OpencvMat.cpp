#include "opencv2/opencv.hpp"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"

void ArmCLImage2OpencvMat(arm_compute::Image &image, cv::Mat &mat)
{
  unsigned char* mat_ptr;
  arm_compute::Window window;
  arm_compute::Iterator in(&image, window);

  window.set(arm_compute::Window::DimX, arm_compute::Window::Dimension(0, mat.cols, 1));
  window.set(arm_compute::Window::DimY, arm_compute::Window::Dimension(0, mat.rows, 1));

  mat_ptr = mat.data;

  arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
  {
    const unsigned char value = *in.ptr();

    *mat_ptr++ = value;
  },
  in);
}
