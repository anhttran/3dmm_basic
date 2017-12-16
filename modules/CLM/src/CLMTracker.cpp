/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2012, Tadas Baltrusaitis, all rights reserved.
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice, 
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions (licence) and the following disclaimer 
//       in the documentation and/or other materials provided with the 
//       distribution.
//     * The name of the author may not be used to endorse or promote products 
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to 
//       and keep in place any licencing terms of those libraries.
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite one of the following works:
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 3D
//       Constrained Local Model for Rigid and Non-Rigid Facial Tracking.
//       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.    
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

#include <CLMTracker.h>

//#include "highgui.h"
//#include "cv.h"
#include <math.h>

using namespace CLMTracker;
using namespace cv;

// Getting a head pose estimate from the currently detected landmarks
Vec6d CLMTracker::GetPoseCLM(CLM& clm_model, double fx, double fy, double cx, double cy, CLMParameters& params)
{
	if(!clm_model.detected_landmarks.empty() && clm_model.params_global[0] != 0)
	{
		double Z = fx / clm_model.params_global[0];
	
		double X = ((clm_model.params_global[4] - cx) * (1.0/fx)) * Z;
		double Y = ((clm_model.params_global[5] - cy) * (1.0/fy)) * Z;
	
		return Vec6d(X, Y, Z, clm_model.params_global[1], clm_model.params_global[2], clm_model.params_global[3]);
	}
	else
	{
		return Vec6d(0,0,0,0,0,0);
	}
}

void GetCentreOfMass(const Mat_<uchar>& mask, double& centreX, double& centreY, const Rect& roi)
{
    if(roi.width == 0)
    {
        Moments baseMoments = moments(mask, true);
        // center of mass x = m_10/m_00, y = m_01/m_00 if using image moments
        centreX = (baseMoments.m10/baseMoments.m00);
        centreY = (baseMoments.m01/baseMoments.m00);
    }
    else
    {
        Moments baseMoments = moments(mask(roi), true);
        // center of mass x = m_10/m_00, y = m_01/m_00 if using image moments

        centreX = (baseMoments.m10/baseMoments.m00) + roi.x;
        centreY = (baseMoments.m01/baseMoments.m00) + roi.y;
    }
}

// If the depth image is provided a head pose estimate can be made more accurate using the actual depth of the face
Vec6d CLMTracker::GetPoseCLM(CLM& clm_model, const cv::Mat_<float> &depth, double fx, double fy, double cx, double cy, CLMParameters& params)
{
	if(depth.empty())
	{
		return GetPoseCLM(clm_model, fx, fy, cx, cy, params);
	}
	else
	{
		// Correct for the actual pose using depth information
		double Z = fx / clm_model.params_global[0];
	
		double X = ((clm_model.params_global[4] - cx) * (1.0/fx)) * Z;
		double Y = ((clm_model.params_global[5] - cy) * (1.0/fy)) * Z;	

		double tx = clm_model.params_global[4];
		double ty = clm_model.params_global[5];

		cv::Mat_<uchar> currentFrameMask = depth > 0;

		// Rough size of the head in mm, for depth correction
		int width = (int)(140 * clm_model.params_global[0]);			
		int height = (int)(133 * clm_model.params_global[0]);			

		Rect roi((int)tx-width/2, (int)ty-width/2, width, height);

		// clamp the ROI
		roi.x = max(roi.x, 0);
		roi.y = max(roi.y, 0);

		roi.x = min(roi.x, depth.cols-1);
		roi.y = min(roi.y, depth.rows-1);

		Vec6d curr_pose(X, Y, Z, clm_model.params_global[1], clm_model.params_global[2], clm_model.params_global[3]);

		// deal with cases where the pose estimate is wildly off
		if(roi.width <= 0) 
		{
			roi.width = depth.cols;
			roi.x = 0;
		}
		if(roi.height <= 0)
		{
			roi.height = depth.rows;
			roi.y = 0;
		}

		if(roi.x + roi.width > depth.cols)
			roi.width = depth.cols - roi.x;
		if(roi.y + roi.height > depth.rows)
			roi.height = depth.rows - roi.y;

		if(sum(currentFrameMask(roi))[0] > 200)
		{
			// Calculate the centers of mass in the mask for a new pose estimate
			double centreX, centreY;
			GetCentreOfMass(currentFrameMask, centreX, centreY, roi);

			// the center of mass gives bad results when shoulder or ponytails are visible
			Z = mean(depth(Rect((int)centreX - 8, (int)centreY - 8, 16, 16)), currentFrameMask(Rect((int)centreX - 8, (int)centreY - 8, 16, 16)))[0] + 100; // Z offset from the surface of the face, as the center of head is not on the surface
			X  = (centreX - cx) * Z / fx;
			Y  = (centreY - cy) * Z / fy; 

			// redefine the pose around witch to sample (only if it's legal)
			if(Z != 100)
			{
				curr_pose[0] = X;
				curr_pose[1] = Y;
				curr_pose[2] = Z;
			}
		}
		return Vec6d(X, Y, Z, clm_model.params_global[1], clm_model.params_global[2], clm_model.params_global[3]);
	}
}

//================================================================================================================
// Landmark detection in videos, need to provide an image and model parameters (default values work well)
// Optionally can provide a bounding box from which to start tracking
//================================================================================================================
// The one that does the actual work
bool CLMTracker::DetectLandmarksInVideo(const Mat_<uchar> &grayscale_image, const Mat_<float> &depth_image, CLM& clm_model, CLMParameters& params)
{
	// First need to decide if the landmarks should be "detected" or "tracked"
	// Detected means running face detection and a larger search area, tracked means initialising from previous step
	// and using a smaller search area

	bool bJanus;
	vector<float> meta_info;

	// Indicating that this is a first detection in video sequence or after restart
	bool initial_detection = !clm_model.tracking_initialised;

	// Perform tracking rather than detection
	bool track_success = false;

	// Only do it if there was a face detection at all
	if(clm_model.tracking_initialised)
	{
		// The area of interest search size will depend if the previous track was successful
		if(!clm_model.detection_success)
		{
			params.window_sizes_current = params.window_sizes_init;
		}
		else
		{
			params.window_sizes_current = params.window_sizes_small;
		}
		
		track_success = clm_model.DetectLandmarks(grayscale_image, depth_image, params, bJanus, meta_info);
		if(!track_success)
		{
			// Make a record that tracking failed
			clm_model.failures_in_a_row++;
		}
		else
		{
			// indicate that tracking is a success
			clm_model.failures_in_a_row = -1;
		}
	}

	// This is used for both detection (if it the tracking has not been initialised yet) or if the tracking failed (however we do this every n frames, for speed)
	// This also has the effect of an attempt to reinitialise just after the tracking has failed, which is useful during large motions
	if(!clm_model.tracking_initialised || (!clm_model.detection_success && clm_model.failures_in_a_row % params.reinit_video_every == 0))
	{
		
		Rect_<double> bounding_box;

		// If the face detector has not been initialised read it in
		if(clm_model.face_detector.empty())
		{
			clm_model.face_detector.load(params.face_detector_location);
		}
		
		bool face_detection_success = CLMTracker::DetectSingleFace(bounding_box, grayscale_image, clm_model.face_detector);

		// Attempt to detect landmarks using the detected face (if unseccessful the detection will be ignored)
		if(face_detection_success)
		{
			// Indicate that tracking has started as a face was detected
			clm_model.tracking_initialised = true;

			// Keep track of old model values so that they can be restored if redetection fails
			Vec6d params_global_init = clm_model.params_global;
			Mat_<double> params_local_init = clm_model.params_local.clone();
			double likelihood_init = clm_model.model_likelihood;
			Mat_<double> detected_landmarks_init = clm_model.detected_landmarks.clone();
			Mat_<double> landmark_likelihoods_init = clm_model.landmark_likelihoods.clone();

			// Use the detected bounding box and empty local parameters
			clm_model.params_local.setTo(0);
			clm_model.pdm.CalcParams(clm_model.params_global, bounding_box, bJanus, clm_model.params_local);	

			// Make sure the search size is large
			params.window_sizes_current = params.window_sizes_init;

			// Do the actual landmark detection (and keep it only if successful)
			bool landmark_detection_success = clm_model.DetectLandmarks(grayscale_image, depth_image, params, bJanus, meta_info);

			// If landmark reinitialisation unsucessful continue from previous estimates
			// if it's initial detection however, do not care if it was successful as the validator might be wrong, so continue trackig
			// regardless
			if(!initial_detection && !landmark_detection_success)
			{

				// Restore previous estimates
				clm_model.params_global = params_global_init;
				clm_model.params_local = params_local_init.clone();
				clm_model.pdm.CalcShape2D(clm_model.detected_landmarks, clm_model.params_local, clm_model.params_global);
				clm_model.model_likelihood = likelihood_init;
				clm_model.detected_landmarks = detected_landmarks_init.clone();
				clm_model.landmark_likelihoods = landmark_likelihoods_init.clone();

				return false;
			}
			else
			{
				clm_model.failures_in_a_row = -1;
				return true;
			}
		}

	}

	// If haven't managet to escape till now - failed
	return false;
	
}

bool CLMTracker::DetectLandmarksInVideo(const Mat_<uchar> &grayscale_image, const Mat_<float> &depth_image, const Rect_<double> bounding_box, CLM& clm_model, CLMParameters& params)
{
	if(bounding_box.width > 0)
	{
		bool bJanus;
		// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
		clm_model.params_local.setTo(0);
		clm_model.pdm.CalcParams(clm_model.params_global, bounding_box, bJanus, clm_model.params_local);		
		// indicate that face was detected so initialisation is not necessary
		clm_model.detection_success = true;
		clm_model.tracking_initialised = true;
	}

	return DetectLandmarksInVideo(grayscale_image, depth_image, clm_model, params);

}

bool CLMTracker::DetectLandmarksInVideo(const Mat_<uchar> &grayscale_image, CLM& clm_model, CLMParameters& params)
{
	return DetectLandmarksInVideo(grayscale_image, Mat_<float>(), clm_model, params);
}

bool CLMTracker::DetectLandmarksInVideo(const Mat_<uchar> &grayscale_image, const Rect_<double> bounding_box, CLM& clm_model, CLMParameters& params)
{
	return DetectLandmarksInVideo(grayscale_image, Mat_<float>(), clm_model, params);
}

//================================================================================================================
// Landmark detection in image, need to provide an image and optionally CLM model together with parameters (default values work well)
// Optionally can provide a bounding box in which detection is performed (this is useful if multiple faces are to be detected in images)
//================================================================================================================

// This is the one where the actual work gets done, other DetectLandmarksInImage calls lead to this one
bool CLMTracker::DetectLandmarksInImage(const Mat_<uchar> &grayscale_image, const Mat_<float> depth_image, const Rect_<double> bounding_box, CLM& clm_model, CLMParameters& params, bool bJanus, vector<float> meta_info, int nView)
{
	// Can have multiple hypotheses
	vector<Vec3d> rotation_hypotheses;

	if(params.multi_view)
	{
		if (nView == 0)
		{
			// Try out different orientation initialisations
			rotation_hypotheses.push_back(Vec3d(0,0,0));
			rotation_hypotheses.push_back(Vec3d(0,0.5236,0));
			rotation_hypotheses.push_back(Vec3d(0,-0.5236,0));
			rotation_hypotheses.push_back(Vec3d(0.5236,0,0));
			rotation_hypotheses.push_back(Vec3d(-0.5236,0,0));
		}
		else if (nView == 1)
		{
			// Try out different orientation initialisations
			rotation_hypotheses.push_back(Vec3d(0,0,0));
			rotation_hypotheses.push_back(Vec3d(0,0.5236,0));
			rotation_hypotheses.push_back(Vec3d(0,-0.5236,0));
		}
		else if (nView == 2)
		{
			// Try out different orientation initialisations
			rotation_hypotheses.push_back(Vec3d(0,-0.5236,0));
			rotation_hypotheses.push_back(Vec3d(0,-0.7854,0));
			rotation_hypotheses.push_back(Vec3d(0,-1.0472,0));
		}
		else if (nView == 3)
		{
			// Try out different orientation initialisations	
			rotation_hypotheses.push_back(Vec3d(0,0.5236,0));
			rotation_hypotheses.push_back(Vec3d(0,0.7854,0));
			rotation_hypotheses.push_back(Vec3d(0,1.0472,0));
		}
		else
		{
			// Assume the face is close to frontal
			rotation_hypotheses.push_back(Vec3d(0,0,0));
		}
	}
	else
	{
		// Assume the face is close to frontal
		rotation_hypotheses.push_back(Vec3d(0,0,0));
	}
	
	// Use the initialisation size for the landmark detection
	params.window_sizes_current = params.window_sizes_init;
	
	// Store the current best estimate
	double best_likelihood;
	Vec6d best_global_parameters;
	Mat_<double> best_local_parameters;
	Mat_<double> best_detected_landmarks;
	Mat_<double> best_landmark_likelihoods;
	bool best_success;

	for(size_t hypothesis = 0; hypothesis < rotation_hypotheses.size(); ++hypothesis)
	{
		// Reset the potentially set clm_model parameters
		clm_model.params_local.setTo(0.0);

		// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
		clm_model.pdm.CalcParams(clm_model.params_global, bounding_box, bJanus, clm_model.params_local, rotation_hypotheses[hypothesis]);
	
		bool success = clm_model.DetectLandmarks(grayscale_image, depth_image, params, bJanus, meta_info);	
				
		if(hypothesis == 0 || best_likelihood < clm_model.model_likelihood)
		{
			best_likelihood = clm_model.model_likelihood;
			best_global_parameters = clm_model.params_global;
			best_local_parameters = clm_model.params_local.clone();
			best_detected_landmarks = clm_model.detected_landmarks.clone();
			best_landmark_likelihoods = clm_model.landmark_likelihoods.clone();
			best_success = success;
		}
	}

	// Store the best estimates in the clm_model
	clm_model.model_likelihood = best_likelihood;
	clm_model.params_global = best_global_parameters;
	clm_model.params_local = best_local_parameters.clone();
	clm_model.detected_landmarks = best_detected_landmarks.clone();
	clm_model.detection_success = best_success;
	clm_model.landmark_likelihoods = best_landmark_likelihoods.clone();
	return best_success;
}

bool CLMTracker::DetectLandmarksInImage(const Mat_<uchar> &grayscale_image, const Mat_<float> depth_image, CLM& clm_model, CLMParameters& params)
{
	Rect_<double> bounding_box;
	bool bJanus;
	vector<float> meta_info;

	// If the face detector has not been initialised read it in
	if(clm_model.face_detector.empty())
	{
		clm_model.face_detector.load(params.face_detector_location);
	}
		
	// Initialise the face detector
	CLMTracker::DetectSingleFace(bounding_box, grayscale_image, clm_model.face_detector);
	
	return DetectLandmarksInImage(grayscale_image, depth_image, bounding_box, clm_model, params, bJanus, meta_info, 0);
}

// Versions not using depth images
bool CLMTracker::DetectLandmarksInImage(const Mat_<uchar> &grayscale_image, const Rect_<double> bounding_box, CLM& clm_model, CLMParameters& params, bool bJanus, vector<float> meta_info, int nView)
{
	return DetectLandmarksInImage(grayscale_image, Mat_<float>(), bounding_box, clm_model, params, bJanus, meta_info, nView);
}

bool CLMTracker::DetectLandmarksInImage(const Mat_<uchar> &grayscale_image, CLM& clm_model, CLMParameters& params)
{
	return DetectLandmarksInImage(grayscale_image, Mat_<float>(), clm_model, params);
}

