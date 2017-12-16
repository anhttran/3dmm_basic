/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include <string>

#include "JanusUtils.h"
#include "CLM.h"
#include "CLMTracker.h"

using std::string;
using std::vector;

namespace CLMJanus
{

  CLMTracker::CLM clm_model;
  CLMTracker::CLMParameters clm_parameters;


  // Forward declaration of file-private functions
  void convert_to_grayscale(const Mat& in, Mat& out);
  int determine_bounding_box(vector<float> annotations, cv::Rect *bbox, int *image_type);


  
  int initialize_lm_detector(string landmark_model)
  {
    std::cout << "Loading landmark detector model\n";
    string empty_face_detect_model = "";
    clm_model.Read(landmark_model, empty_face_detect_model);

    clm_parameters = CLMTracker::CLMParameters();

    // No need to validate detections, as we're not doing tracking
    clm_parameters.validate_detections = false;
    clm_parameters.model_location = landmark_model;
    clm_parameters.quiet_mode = true;

    // In the Wild settings
    clm_parameters.window_sizes_init = vector<int>(3);
    clm_parameters.window_sizes_init[0] = 15;
    clm_parameters.window_sizes_init[1] = 15;
    clm_parameters.window_sizes_init[2] = 15;
    clm_parameters.sigma = 2;
    clm_parameters.reg_factor = 25;
    clm_parameters.weight_factor = 5;
    clm_parameters.num_optimisation_iteration = 10;

    std::cout << "Landmark detector model loaded\n";
  }


  // NOTE-- This code is labeling LEFT/RIGHT eye in the OPPOSITE way that Janus metadata does. 
  //           i.e. Janus metadata "right eye" = subject's right. Our "right eye" = right from perspective of viewer
  //        This is regrettable, but this is copied from original multicomp implementation
  enum JanusAnnotation {
    JANUS_FACE_X = 0,
    JANUS_FACE_Y,
    JANUS_FACE_WIDTH,
    JANUS_FACE_HEIGHT,
    JANUS_LEFT_EYE_X,
    JANUS_LEFT_EYE_Y,
    JANUS_RIGHT_EYE_X,
    JANUS_RIGHT_EYE_Y,
    JANUS_NOSE_BASE_X,
    JANUS_NOSE_BASE_Y
  };

  int detect_landmarks(cv::Mat img, vector<float> annotations, vector<cv::Point> *out_landmarks, float *out_confidence)
  {
    std::cout << "Starting landmark detection\n";

    // First, we need to determine bounding box based on annotations
    // (Also perform sanity check on annotations)
    cv::Rect bbox;
    int image_type;

    if (annotations.size() != 10)
      return 1;

    if (annotations[JANUS_FACE_X] == 0 &&
	annotations[JANUS_FACE_Y] == 0 &&
	annotations[JANUS_FACE_WIDTH] == 0 &&
	annotations[JANUS_FACE_HEIGHT] == 0 &&
	annotations[JANUS_RIGHT_EYE_X] == 0 &&
	annotations[JANUS_RIGHT_EYE_Y] == 0 &&
	annotations[JANUS_LEFT_EYE_X] == 0 &&
	annotations[JANUS_LEFT_EYE_Y] == 0 &&
	annotations[JANUS_NOSE_BASE_X] == 0 &&
	annotations[JANUS_NOSE_BASE_Y] == 0) {

      std::cout << "Error: no annotations at all are set, skipping entry" << std::endl;
      return 1;
    }

    if (determine_bounding_box(annotations, &bbox, &image_type) != 0)
      return 1;

    std::cout << "Found bounding box, continuing on with bbox = " << bbox << std::endl;

    // Convert to grayscale
    cv::Mat grayscale_image;
    convert_to_grayscale(img, grayscale_image);

    int n = clm_model.pdm.NumberOfPoints();
    float normalized_error;
    if (image_type == 1 || image_type == 4) {
      // Set default value for nviews
      int nviews = 1;

      // Override default in certain cases
      if (annotations[JANUS_FACE_WIDTH] != 0.0 &&
	  annotations[JANUS_RIGHT_EYE_X] < annotations[JANUS_FACE_X] + annotations[JANUS_FACE_WIDTH]/3) {
	nviews = 3;
      }
      else if (annotations[JANUS_FACE_WIDTH] != 0.0 &&
	       annotations[JANUS_LEFT_EYE_X] > annotations[JANUS_FACE_X] + annotations[JANUS_FACE_WIDTH] * 2/3) {
	nviews = 2;
      }

      CLMTracker::DetectLandmarksInImage(grayscale_image, bbox, clm_model, clm_parameters, /*bJanus=*/true, annotations, nviews);

      float nose_x = clm_model.detected_landmarks.at<double>(33);
      float nose_y = clm_model.detected_landmarks.at<double>(33+n);
      float right_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(42) + clm_model.detected_landmarks.at<double>(45) );
      float right_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(42+n) + clm_model.detected_landmarks.at<double>(45+n) );
      float left_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(36) + clm_model.detected_landmarks.at<double>(39) );
      float left_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(36+n) + clm_model.detected_landmarks.at<double>(39+n) );

      float interocular_distance = sqrt( (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) *
					 (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) + 
					 (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) *
					 (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) );
      
      normalized_error = (sqrt( ( annotations[JANUS_RIGHT_EYE_X] - right_eye_x) * ( annotations[JANUS_RIGHT_EYE_X] - right_eye_x) +
				( annotations[JANUS_RIGHT_EYE_Y] - right_eye_y) * ( annotations[JANUS_RIGHT_EYE_Y] - right_eye_y) ) +
			  sqrt( ( annotations[JANUS_NOSE_BASE_X] - nose_x) * ( annotations[JANUS_NOSE_BASE_X] - nose_x) +
				( annotations[JANUS_NOSE_BASE_Y] - nose_y) * ( annotations[JANUS_NOSE_BASE_Y] - nose_y) ) +
			  sqrt( ( annotations[JANUS_LEFT_EYE_X] - left_eye_x) * ( annotations[JANUS_LEFT_EYE_X] - left_eye_x)  +
				( annotations[JANUS_LEFT_EYE_Y] - left_eye_y) * ( annotations[JANUS_LEFT_EYE_Y] - left_eye_y) )
			  ) / 3 / interocular_distance;
    }
    else if (image_type == 7 || image_type == 8) {
      // Set default value for nviews
      int nviews = 1;

      // Override default in certain cases
      if (annotations[JANUS_FACE_WIDTH] != 0.0 &&
	  annotations[JANUS_RIGHT_EYE_X] < annotations[JANUS_FACE_X] + annotations[JANUS_FACE_WIDTH]/3) {
	nviews = 3;
      }
      else if (annotations[JANUS_FACE_WIDTH] != 0.0 &&
	       annotations[JANUS_LEFT_EYE_X] > annotations[JANUS_FACE_X] + annotations[JANUS_FACE_WIDTH] * 2/3) {
	nviews = 2;
      }

      CLMTracker::DetectLandmarksInImage(grayscale_image, bbox, clm_model, clm_parameters, /*bJanus=*/true, annotations, nviews);

      float nose_x = clm_model.detected_landmarks.at<double>(33);
      float nose_y = clm_model.detected_landmarks.at<double>(33+n);
      float right_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(42) + clm_model.detected_landmarks.at<double>(45) );
      float right_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(42+n) + clm_model.detected_landmarks.at<double>(45+n) );
      float left_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(36) + clm_model.detected_landmarks.at<double>(39) );
      float left_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(36+n) + clm_model.detected_landmarks.at<double>(39+n) );

      float interocular_distance = sqrt( (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) *
					 (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) + 
					 (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) *
					 (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) );

      normalized_error = (sqrt( ( annotations[JANUS_RIGHT_EYE_X] - right_eye_x) * ( annotations[JANUS_RIGHT_EYE_X] - right_eye_x) + 
				( annotations[JANUS_RIGHT_EYE_Y] - right_eye_y) * ( annotations[JANUS_RIGHT_EYE_Y] - right_eye_y) ) +
			  sqrt( ( annotations[JANUS_LEFT_EYE_X] - left_eye_x) * ( annotations[JANUS_LEFT_EYE_X] - left_eye_x)  +
				( annotations[JANUS_LEFT_EYE_Y] - left_eye_y) * ( annotations[JANUS_LEFT_EYE_Y] - left_eye_y) )
			  ) / 2 / interocular_distance;
    }
    else if (image_type == 2 || image_type == 5) {
      CLMTracker::DetectLandmarksInImage(grayscale_image, bbox, clm_model, clm_parameters, /*bJanus=*/true, annotations, 2);

      float nose_x = clm_model.detected_landmarks.at<double>(33);
      float nose_y = clm_model.detected_landmarks.at<double>(33+n);
      float right_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(42) + clm_model.detected_landmarks.at<double>(45) );
      float right_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(42+n) + clm_model.detected_landmarks.at<double>(45+n) );
      float left_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(36) + clm_model.detected_landmarks.at<double>(39) );
      float left_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(36+n) + clm_model.detected_landmarks.at<double>(39+n) );

      float interocular_distance = sqrt( (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) *
					 (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) +
					 (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) *
					 (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) );

      normalized_error = (sqrt( ( annotations[JANUS_NOSE_BASE_X] - nose_x) * ( annotations[JANUS_NOSE_BASE_X] - nose_x) +
				( annotations[JANUS_NOSE_BASE_Y] - nose_y) * ( annotations[JANUS_NOSE_BASE_Y] - nose_y) ) +
			  sqrt( ( annotations[JANUS_LEFT_EYE_X] - left_eye_x) * ( annotations[JANUS_LEFT_EYE_X] - left_eye_x) +
				( annotations[JANUS_LEFT_EYE_Y] - left_eye_y) * ( annotations[JANUS_LEFT_EYE_Y] - left_eye_y) )
			  ) / 2 / interocular_distance;

    }
    else if (image_type == 3 || image_type == 6) {
      CLMTracker::DetectLandmarksInImage(grayscale_image, bbox, clm_model, clm_parameters, /*bJanus=*/true, annotations, 3);

      float nose_x = clm_model.detected_landmarks.at<double>(33);
      float nose_y = clm_model.detected_landmarks.at<double>(33+n);
      float right_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(42) + clm_model.detected_landmarks.at<double>(45) );
      float right_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(42+n) + clm_model.detected_landmarks.at<double>(45+n) );
      float left_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(36) + clm_model.detected_landmarks.at<double>(39) );
      float left_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(36+n) + clm_model.detected_landmarks.at<double>(39+n) );
			
      float interocular_distance = sqrt( (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) *
					 (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) +
					 (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) *
					 (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) );

      normalized_error = (sqrt( ( annotations[JANUS_RIGHT_EYE_X] - right_eye_x) * ( annotations[JANUS_RIGHT_EYE_X] - right_eye_x) +
				( annotations[JANUS_RIGHT_EYE_Y] - right_eye_y) * ( annotations[JANUS_RIGHT_EYE_Y] - right_eye_y) ) +
			  sqrt( ( annotations[JANUS_NOSE_BASE_X] - nose_x) * ( annotations[JANUS_NOSE_BASE_X] - nose_x) +
				( annotations[JANUS_NOSE_BASE_Y] - nose_y) * ( annotations[JANUS_NOSE_BASE_Y] - nose_y) )
			  ) / 2 / interocular_distance;
			  
    }
    else {
      std::cout << "Error: Landmark Detection -- Unknown image type: " << image_type << std::endl;
      return 1;
    }

    // Now it's time to create output landmark vector
    for (int i = 0; i < 68; ++i) {
      out_landmarks->push_back( cv::Point(clm_model.detected_landmarks.at<double>(i), clm_model.detected_landmarks.at<double>(i+n)) );
    }
    *out_confidence = normalized_error;
    std::cout << "Ending landmark detection\n";

    return 0;
  }


  int determine_bounding_box(vector<float> annotations, cv::Rect *bbox, int *image_type) {

    if (annotations[JANUS_FACE_WIDTH] != 0.0 && annotations[JANUS_FACE_HEIGHT] != 0.0) {

      if (annotations[JANUS_RIGHT_EYE_X] != 0.0 &&
	  annotations[JANUS_LEFT_EYE_X]  != 0.0 &&
	  annotations[JANUS_NOSE_BASE_X] != 0.0) {

	if (annotations[JANUS_FACE_X] + annotations[JANUS_FACE_WIDTH] <= annotations[JANUS_LEFT_EYE_X])
	  {
	    std::cout << "ERROR: Bad Annotation for landmark detection -- the left eye does not lay within the face boundaries";
	    return 1;
	  }

	float x = annotations[JANUS_LEFT_EYE_X];
	float y = 0.5 * (annotations[JANUS_LEFT_EYE_Y] + annotations[JANUS_RIGHT_EYE_Y]);
	float w = annotations[JANUS_RIGHT_EYE_X] - x;
	float h = annotations[JANUS_NOSE_BASE_Y] - y;

	*bbox = cv::Rect_<float>(x, y, w, h);

	std::cout << "1) rect = " << *bbox << std::endl;

	*image_type = 1;
      }
      else if (annotations[JANUS_RIGHT_EYE_X] != 0.0 &&
	       annotations[JANUS_LEFT_EYE_X] != 0.0) {

	float x = annotations[JANUS_LEFT_EYE_X];
	float y = 0.5 * (annotations[JANUS_LEFT_EYE_Y] + annotations[JANUS_RIGHT_EYE_Y]);
	float w = annotations[JANUS_RIGHT_EYE_X] - x;
	float h = 0.85 * w;

	*bbox = cv::Rect_<float>(x, y, w, h);
	*image_type = 7;

	std::cout << "2) rect = " << *bbox << std::endl;
      }
      else if (annotations[JANUS_LEFT_EYE_X] != 0.0 &&
	       annotations[JANUS_NOSE_BASE_X] != 0.0) {

	float x = 2 * annotations[JANUS_LEFT_EYE_X] - annotations[JANUS_NOSE_BASE_X];
	float y = annotations[JANUS_LEFT_EYE_Y];
	float w = 2 * (annotations[JANUS_NOSE_BASE_X] - annotations[JANUS_LEFT_EYE_X]);
	float h = annotations[JANUS_NOSE_BASE_Y] - y;

	*bbox = cv::Rect_<float>(x, y, w, h);
	*image_type = 2;

	std::cout << "3) rect = " << *bbox << std::endl;
      }
      else if (annotations[JANUS_RIGHT_EYE_X] != 0.0 &&
	       annotations[JANUS_NOSE_BASE_X] != 0.0) {

	float x = annotations[JANUS_NOSE_BASE_X];
	float y = annotations[JANUS_RIGHT_EYE_Y];
	float w = 2 * (annotations[JANUS_RIGHT_EYE_X] - x);
	float h = annotations[JANUS_NOSE_BASE_Y] - y;

	*bbox = cv::Rect_<float>(x, y, w, h);
	*image_type = 3;
	std::cout << "4) rect = " << *bbox << std::endl;
      }
      else {
	std::cout << "ERROR: Bad Annotation for landmark detection -- none of the annotation-branches was taken";
	return 1;
      }
    } else {
      // (In this branch no face width and/or face height have been provided

      if (annotations[JANUS_RIGHT_EYE_X] != 0.0 &&
	  annotations[JANUS_LEFT_EYE_X]  != 0.0 &&
	  annotations[JANUS_NOSE_BASE_X] != 0.0) {

	float x = annotations[JANUS_LEFT_EYE_X];
	float y = 0.5 * (annotations[JANUS_LEFT_EYE_Y] + annotations[JANUS_RIGHT_EYE_Y]);
	float w = annotations[JANUS_RIGHT_EYE_X] - x;
	float h = annotations[JANUS_NOSE_BASE_Y] - y;

	*bbox = cv::Rect_<float>(x, y, w, h);
	*image_type = 4;
	std::cout << "5) rect = " << *bbox << std::endl;
      }
      else if (annotations[JANUS_RIGHT_EYE_X] != 0.0 &&
	       annotations[JANUS_LEFT_EYE_X] != 0.0) {

	float x = annotations[JANUS_LEFT_EYE_X];
	float y = 0.5 * (annotations[JANUS_LEFT_EYE_Y] + annotations[JANUS_RIGHT_EYE_Y]);
	float w = annotations[JANUS_RIGHT_EYE_X] - x;
	float h = 0.85 * w;

	*bbox = cv::Rect_<float>(x, y, w, h);
	*image_type = 8;
	std::cout << "6) rect = " << *bbox << std::endl;
      }
      else if (annotations[JANUS_LEFT_EYE_X] != 0.0 &&
	       annotations[JANUS_NOSE_BASE_X] != 0.0) {

	float x = 2 * annotations[JANUS_LEFT_EYE_X] - annotations[JANUS_NOSE_BASE_X];
	float y = annotations[JANUS_LEFT_EYE_Y];
	float w = 2 * (annotations[JANUS_NOSE_BASE_X] - annotations[JANUS_LEFT_EYE_X]);
	float h = annotations[JANUS_NOSE_BASE_Y] - y;

	*bbox = cv::Rect_<float>(x, y, w, h);
	*image_type = 5;
	std::cout << "7) rect = " << *bbox << std::endl;
      }
      else if (annotations[JANUS_RIGHT_EYE_X] != 0.0 &&
	       annotations[JANUS_NOSE_BASE_X] != 0.0) {

	float x = annotations[JANUS_NOSE_BASE_X];
	float y = annotations[JANUS_RIGHT_EYE_Y];
	float w = 2 * (annotations[JANUS_RIGHT_EYE_X] - annotations[JANUS_NOSE_BASE_X]);
	float h = annotations[JANUS_NOSE_BASE_Y] - y;

	*bbox = cv::Rect_<float>(x, y, w, h);
	*image_type = 6;
	std::cout << "8) rect = " << *bbox << std::endl;
      }
      else {
	std::cout << "ERROR: Bad Annotation for landmark detection -- none of the annotation-branches was taken (and face-width=0.0)";
	return 1;
      }
    }

    return 0;
  }


  void convert_to_grayscale(const Mat& in, Mat& out)
  {
    if(in.channels() == 3)
      {
	// Make sure it's in a correct format
	if(in.depth() != CV_8U)
	  {
	    if(in.depth() == CV_16U)
	      {
		Mat tmp = in / 256;
		tmp.convertTo(tmp, CV_8U);
		cvtColor(tmp, out, CV_BGR2GRAY);
	      }
	  }
	else
	  {
	    cvtColor(in, out, CV_BGR2GRAY);
	  }
      }
    else
      {
	if(in.depth() == CV_16U)
	  {
	    Mat tmp = in / 256;
	    out = tmp.clone();
	  }
	else
	  {
	    out = in.clone();
	  }
      }
  }
}
