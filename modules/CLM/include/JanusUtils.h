/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#ifndef __JANUS_UTILS_h__
#define __JANUS_UTILS_h__

#include <string>
#include <vector>

#include <cv.h>
#include <highgui.h>

namespace CLMJanus
{

  int initialize_lm_detector(std::string landmark_model);
  int detect_landmarks(cv::Mat img, std::vector<float> annotations, std::vector<cv::Point> *out_landmarks, float *out_confidence);

}


#endif
