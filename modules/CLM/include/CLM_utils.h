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


//  Header for all external CLM methods of interest to the user
//
//
//  Tadas Baltrusaitis
//  28/03/2014

#ifndef __CLM_UTILS_h_
#define __CLM_UTILS_h_

//#include "cv.h"
//#include "highgui.h"

#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <fstream>
#include <iostream>

#include "CLM.h"

using namespace std;
using namespace cv;

namespace CLMTracker
{
	typedef struct AlgParam {
		double pose_param, pose_certainty, pose_thresh, ac_certainty, ac_thresh, rc_certainty,eye_err_thresh, hogErrMaxThresh, hogErrMinThresh, maxHOG_Weight;
		bool   pose_normalize, checkLMVisible, checkHOGErr, copyBetterPose, checkPrevPose, fixPose,keepBox;

		double deformCertThresh1, deformCertThresh2, deformAngleThresh1, deformAngleThresh2, deformHOGThresh1, deformHOGThresh2, deformGap;
		bool   deformLM, markLM, inferLR;

		AlgParam() {
			pose_param = 0.6;
			ac_certainty = -0.5;
			ac_thresh = 30;
			eye_err_thresh = 0.25;
			pose_certainty = -0.4;
			pose_thresh = 45;

			hogErrMaxThresh = 70;
			hogErrMinThresh = 20;
			maxHOG_Weight   = 0.5f;

			rc_certainty = -0.4;

			fixPose = true;
			checkLMVisible = false;
			checkHOGErr    = true;
			copyBetterPose = false;
			checkPrevPose = false;
			pose_normalize = false;
			keepBox = false;
			
			markLM = false;
			inferLR = false;
			deformLM = false;
			deformAngleThresh1 = 90;
			deformCertThresh1  = -0.7;
			deformHOGThresh1   = 35;
			deformAngleThresh2 = 30;
			deformCertThresh1  = -0.5;
			deformHOGThresh1   = 50;
			deformGap = 5;
		}
	} AlgParam;
	//===========================================================================	
	// Defining a set of useful utility functions to be used within CLM

	//=============================================================================================
	// Helper functions for parsing the inputs
	//=============================================================================================
	void get_video_input_output_params(vector<string> &input_video_file, vector<string> &depth_dir,
		vector<string> &output_pose_file, vector<string> &output_video_file, vector<string> &output_features_file, vector<string> &model_output, vector<string> &arguments);
	void get_video_metadata(vector<vector<float> > &input_meta_info,vector<int> &input_meta_frames,vector<Rect_<double> > &input_bounding_boxes, vector<int> &input_image_types, vector<string> &arguments);

	void get_camera_params(int &device, float &fx, float &fy, float &cx, float &cy, vector<string> &arguments);
	void get_alg_params(AlgParam &al_param, vector<string> &arguments);

	void get_basel_params(vector<string> &arguments, string &baselFile);
	void get_run_params(vector<string> &arguments, string &startFFile, string &badIn, int &fromIndex, int &toIndex);
	bool loadLM(string lmPath, string cfdPath, string posePath, cv::Mat_<double> &oriLMs, double &cfd, cv::Vec6d &poseCLM);
	void get_image_input_output_params(vector<string> &input_image_files, vector<string> &input_depth_files, vector<string> &output_feature_files, vector<string> &output_image_files, vector<string> &output_model_files, vector<string> &output_lm_files, vector<string> &output_pose_files,
		vector<Rect_<double> > &input_bounding_boxes, bool &bJanus, vector<vector<float> > &input_meta_info, vector<int> &input_image_types, vector<string> &arguments, string &opath, string &olist, vector<string> &inputLMFiles, vector<int> &similarFile, bool &continueWork, string &refDir);
	void get_image_alg_params(vector<string> &arguments, float &mstep, float &threshSize, float &speed, bool &preOpt, int &numTri);	
	//===========================================================================
	// Fast patch expert response computation (linear model across a ROI) using normalised cross-correlation
	//===========================================================================
	// This is a modified version of openCV code that allows for precomputed dfts of templates and for precomputed dfts of an image
	// _img is the input img, _img_dft it's dft (optional), _integral_img the images integral image (optional), squared integral image (optional), 
	// templ is the template we are convolving with, templ_dfts it's dfts at varying windows sizes (optional),  _result - the output, method the type of convolution
	void matchTemplate_m( InputArray _img, Mat& _img_dft, Mat& _integral_img, Mat& _integral_img_sq, InputArray _templ, map<int, Mat>& _templ_dfts, cv::Mat_<float>& result, int method );

	//===========================================================================
	// Point set and landmark manipulation functions
	//===========================================================================
	// Using Kabsch's algorithm for aligning shapes
	//This assumes that align_from and align_to are already mean normalised
	Matx22d AlignShapesKabsch2D(const Mat_<double>& align_from, const Mat_<double>& align_to );

	//=============================================================================
	// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
	Matx22d AlignShapesWithScale(cv::Mat_<double>& src, cv::Mat_<double> dst);

	//===========================================================================
	// Visualisation functions
	//===========================================================================
	void Project(Mat_<double>& dest, const Mat_<double>& mesh, Size size, double fx, double fy, double cx, double cy);
	void DrawBox(Mat image, Vec6d pose, Scalar color, int thickness, float fx, float fy, float cx, float cy);

	void Draw(cv::Mat img, const Mat& shape2D, Mat& visibilities);
	void Draw(cv::Mat img, CLM& clm_model);

	//===========================================================================
	// Angle representation conversion helpers
	//===========================================================================
	Matx33d Euler2RotationMatrix(const Vec3d& eulerAngles);

	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	Vec3d RotationMatrix2Euler(const Matx33d& rotation_matrix);

	Vec3d Euler2AxisAngle(const Vec3d& euler);

	Vec3d AxisAngle2Euler(const Vec3d& axis_angle);

	Matx33d AxisAngle2RotationMatrix(const Vec3d& axis_angle);

	Vec3d RotationMatrix2AxisAngle(const Matx33d& rotation_matrix);

	//============================================================================
	// Face detection helpers
	//============================================================================

	// Face detection using Haar cascade classifier
	bool DetectFaces(vector<Rect_<double> >& o_regions, const Mat_<uchar>& intensity);
	bool DetectFaces(vector<Rect_<double> >& o_regions, const Mat_<uchar>& intensity, CascadeClassifier& classifier);
	bool DetectSingleFace(Rect_<double>& o_region, const Mat_<uchar>& intensity, CascadeClassifier& classifier);

	//============================================================================
	// Matrix reading functionality
	//============================================================================

	// Reading a matrix written in a binary format
	void ReadMatBin(std::ifstream& stream, Mat &output_mat);

	// Reading in a matrix from a stream
	void ReadMat(std::ifstream& stream, Mat& output_matrix);

	// Skipping comments (lines starting with # symbol)
	void SkipComments(std::ifstream& stream);

}
#endif
