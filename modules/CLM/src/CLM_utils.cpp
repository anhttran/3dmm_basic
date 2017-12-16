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

#include "CLM_utils.h"

#include <math.h>
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#include <opencv2/core/internal.hpp>
#include <iostream>
#include <sstream>

using namespace boost::filesystem;

using namespace cv;
using namespace std;

class JanusMetaData
{ 
private:
	
	int template_id, subject_id, frame, media_id, sighting_id; 
	float face_x, face_y, face_width, face_height, right_eye_x, right_eye_y, left_eye_x, left_eye_y, nose_base_x, nose_base_y;
	string file;

public:
		
	JanusMetaData() {template_id = subject_id = frame = media_id = sighting_id = 0; 
					face_x = face_y = face_width = face_height = right_eye_x = right_eye_y = left_eye_x = left_eye_y = nose_base_x = nose_base_y = 0.0;
					file="";}
	
	void Load(std::ifstream &f, bool shortForm = false);
	
	int GetTemplateId() { return template_id; } 
	int GetSubjectId() { return subject_id; } 
	int GetMediaId() { return media_id; } 
	int GetFrame() { return frame; } 
	string &GetFile() { return file; } 
	float GetFaceX() { return face_x; } 
	float GetFaceY() { return face_y; } 
	float GetFaceWidth() { return face_width; } 
	float GetFaceHeight() { return face_height; } 
	float GetRightEyeX() { return right_eye_x; } 
	float GetRightEyeY() { return right_eye_y; } 
	float GetLeftEyeX() { return left_eye_x; } 
	float GetLeftEyeY() { return left_eye_y; } 
	float GetNoseBaseX() { return nose_base_x; } 
	float GetNoseBaseY() { return nose_base_y; } 
};

void JanusMetaData::Load(std::ifstream &f, bool shortForm)
{    
	char ch;
	string line, temp;
	
	getline(f, line);
	istringstream ins(line);

	if (!shortForm) {
		ins >> template_id >> ch >> subject_id >> ch;
	}
	getline(ins, file, ',');
	getline(ins, temp, ',');
	media_id = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	sighting_id = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	frame = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	face_x = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	face_y = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	face_width = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	face_height = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	left_eye_x = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	left_eye_y = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	right_eye_x = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	right_eye_y = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	nose_base_x = (int)atof(temp.c_str());
	getline(ins, temp, ',');
	nose_base_y = (int)atof(temp.c_str());
} 

namespace CLMTracker
{
	bool loadLM(string lmPath, string cfdPath, string posePath, cv::Mat_<double> &oriLMs, double &cfd, cv::Vec6d &poseCLM){

		// parse the -fmeta directory by reading in it
		path meta_file (lmPath); 

		if (exists(lmPath))   
		{
			std::cout << "cfd " << cfdPath << std::endl;
			std::ifstream in_meta(meta_file.string());
			oriLMs = cv::Mat_<double>::zeros(2*68,1);
			int cnt = 0;
			string line;
			//getline(in_meta, line);
			//getline(in_meta, line);
			//getline(in_meta, line);
			while(!in_meta.eof())
			{
				getline(in_meta, line);
				if (line.length() > 4) {
					istringstream ins(line);
					ins >> oriLMs.at<double>(cnt) >> oriLMs.at<double>(cnt+68);
					cnt++;
				}	
			}
			in_meta.close();

			if (exists(cfdPath))   
			{
				std::cout << "cfd " << cfdPath << std::endl;
				std::ifstream in_cfd(cfdPath);
				string line;
				getline(in_cfd, line);
				istringstream ins(line);
				ins >> cfd;
				in_cfd.close();
				poseCLM = Vec6d(0,0,0,0,0,0);
				/*if (exists(posePath))   
				{
					std::ifstream in_pose(posePath);
					string lineP, temp;
					getline(in_pose, lineP);
					istringstream ins2(lineP);
					ins2 >> poseCLM(0) >> poseCLM(1) >> poseCLM(2) >> poseCLM(3) >> poseCLM(4) >> poseCLM(5) ;
					std::cout << "pose " << poseCLM(0) << "," << poseCLM(1) << "," << poseCLM(2) << "," << poseCLM(3) << "," << poseCLM(4) << "," << poseCLM(5) << std::endl;
					in_pose.close();
				}*/
				return true;
			}
			else return false;

		}
		else return false;
	}


// Extracting the following command line arguments -f, -fd, -op, -of, -ov (and possible ordered repetitions)
void get_video_input_output_params(vector<string> &input_video_files, vector<string> &depth_dirs,
	vector<string> &output_pose_files, vector<string> &output_video_files, vector<string> &output_features_files, vector<string> &model_output, vector<string> &arguments)
{
	bool* valid = new bool[arguments.size()];

	for(size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	string root = "";
	// First check if there is a root argument (so that videos and outputs could be defined more easilly)
	for(size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-root") == 0) 
		{                    
			root = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
	}

	for(size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-f") == 0) 
		{                    
			input_video_files.push_back(root + arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if (arguments[i].compare("-fd") == 0) 
		{                    
			depth_dirs.push_back(root + arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;		
			i++;
		} 
		else if (arguments[i].compare("-op") == 0)
		{
			output_pose_files.push_back(root + arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;
			i++;
		} 
		else if (arguments[i].compare("-of") == 0)
		{
			output_features_files.push_back(root + arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;
			i++;
		} 
		else if (arguments[i].compare("-ov") == 0)
		{
			output_video_files.push_back(root + arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;
			i++;
		} 
		if (arguments[i].compare("-mo") == 0) 
		{                    
			model_output.push_back(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}	
		else if (arguments[i].compare("-help") == 0)
		{
			cout << "Input output files are defined as: -f <infile> -fd <indepthdir> -op <outpose> -of <outfeatures> -ov <outvideo>\n"; // Inform the user of how to use the program				
		}
	}

	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}

}

void get_camera_params(int &device, float &fx, float &fy, float &cx, float &cy, vector<string> &arguments)
{
	bool* valid = new bool[arguments.size()];

	for(size_t i=0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-fx") == 0) 
		{                    
			fx = stof(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if (arguments[i].compare("-fy") == 0) 
		{                    
			fy = stof(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;		
			i++;
		} 
		else if (arguments[i].compare("-cx") == 0)
		{
			cx = stof(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;
			i++;
		} 
		else if (arguments[i].compare("-cy") == 0)
		{
			cy = stof(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;
			i++;
		}
		else if (arguments[i].compare("-device") == 0)
		{
			device = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;
			i++;
		}
		else if (arguments[i].compare("-help") == 0)
		{
			cout << "Camera parameters are defined as: -device <webcam number> -fx <float focal length x> -fy <float focal length y> -cx <float optical center x> -cy <float optical center y> "  << endl; // Inform the user of how to use the program				
		}
	}

	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}
}

void get_alg_params(AlgParam &al_param, vector<string> &arguments)
{
	bool* valid = new bool[arguments.size()];

	for(size_t i=0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-pp") == 0) 
		{                    
			al_param.pose_param = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}	
		if (arguments[i].compare("-pt") == 0) 
		{                    
			al_param.pose_thresh = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-pc") == 0) 
		{                    
			al_param.pose_certainty = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-pn") == 0) 
		{                    
			al_param.pose_normalize = (arguments[i + 1][0] == '1');
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}	
		if (arguments[i].compare("-ac") == 0) 
		{                    
			al_param.ac_certainty = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}	
		if (arguments[i].compare("-at") == 0) 
		{                    
			al_param.ac_thresh = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}	
		if (arguments[i].compare("-ee") == 0) 
		{                    
			al_param.eye_err_thresh = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}	
		if (arguments[i].compare("-hogmax") == 0) 
		{                    
			al_param.hogErrMaxThresh  = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-hogmin") == 0) 
		{                    
			al_param.hogErrMinThresh   = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-ckVis") == 0) 
		{                    
			al_param.checkLMVisible = (arguments[i + 1][0] != '0');
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-ckLM") == 0) 
		{                    
			al_param.checkHOGErr = (arguments[i + 1][0] != '0');
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-cpPose") == 0) 
		{                    
			al_param.copyBetterPose = (arguments[i + 1][0] != '0');
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-ckPrev") == 0) 
		{                    
			al_param.checkPrevPose = (arguments[i + 1][0] != '0');
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-fixPose") == 0) 
		{                    
			al_param.fixPose = (arguments[i + 1][0] != '0');
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-maxHOGW") == 0) 
		{                    
			al_param.maxHOG_Weight = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		
		if (arguments[i].compare("-deformAng1") == 0) 
		{                    
			al_param.deformAngleThresh1   = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-deformAng2") == 0) 
		{                    
			al_param.deformAngleThresh2   = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-deformCert1") == 0) 
		{                    
			al_param.deformCertThresh1   = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-deformCert2") == 0) 
		{                    
			al_param.deformCertThresh2   = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-deformHOG1") == 0) 
		{                    
			al_param.deformHOGThresh1   = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-deformHOG2") == 0) 
		{                    
			al_param.deformHOGThresh2   = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-deform") == 0) 
		{                    
			al_param.deformLM = (arguments[i + 1][0] != '0');
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-deformGap") == 0) 
		{                    
			al_param.deformGap   = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-markLM") == 0) 
		{                    
			al_param.markLM = (arguments[i + 1][0] != '0');
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		if (arguments[i].compare("-inferLR") == 0) 
		{                    
			al_param.inferLR = (arguments[i + 1][0] != '0');
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
	}

	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}
}

void get_basel_params(vector<string> &arguments, string &baselFile){
	baselFile = "BaselFace.dat";
	bool* valid = new bool[arguments.size()];

	for(size_t i=0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-basel") == 0) 
		{                    
			baselFile = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
	}

	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}	
}

void get_run_params(vector<string> &arguments, string &startFFile, string &badIn, int &fromIndex, int &toIndex){
	fromIndex = 0;
	toIndex = -1;
	startFFile = "startF.txt";
	badIn = "badIn.txt";
	bool* valid = new bool[arguments.size()];

	for(size_t i=0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-startF") == 0) 
		{                    
			startFFile = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}

		if (arguments[i].compare("-from") == 0) 
		{                    
			fromIndex = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}

		if (arguments[i].compare("-to") == 0) 
		{                    
			toIndex = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}                
		if (arguments[i].compare("-badIn") == 0)
                {
                        badIn = arguments[i + 1];
                        valid[i] = false;
                        valid[i+1] = false;
                        i++;
                }

	}

	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}	
}

void get_video_metadata(vector<vector<float> > &input_meta_info,vector<int> &input_meta_frames,vector<Rect_<double> > &input_bounding_boxes, vector<int> &input_image_types, vector<string> &arguments){
	bool* valid = new bool[arguments.size()];
	int eye_type = 0;

	for(size_t i=0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-fmeta") == 0) 
		{
			// parse the -fmeta directory by reading in it
			path meta_file (arguments[i+1]); 

			if (exists(meta_file))   
			{
				std::ifstream in_meta(meta_file.string());
				string str;
				JanusMetaData data; 
				vector<JanusMetaData> vdata;
				int cnt = 1;
				while(!in_meta.eof())
				{
					data.Load(in_meta,true);				
					if (cnt != 0) vdata.push_back(data);
					if (in_meta.eof()) break; 
					cnt++;
				}
				in_meta.close();

				vector<JanusMetaData>::iterator it; 
				for (it = vdata.begin(); it != vdata.end(); it++) 
				{
					vector<float> meta_info;
					meta_info.push_back((*it).GetFaceX());
					meta_info.push_back((*it).GetFaceY());
					meta_info.push_back((*it).GetFaceWidth());
					meta_info.push_back((*it).GetFaceHeight());
					meta_info.push_back((*it).GetRightEyeX());
					meta_info.push_back((*it).GetRightEyeY());
					meta_info.push_back((*it).GetLeftEyeX());
					meta_info.push_back((*it).GetLeftEyeY());
					meta_info.push_back((*it).GetNoseBaseX());
					meta_info.push_back((*it).GetNoseBaseY());

					if((*it).GetFaceWidth() != 0.0 && (*it).GetFaceHeight() != 0.0)
					{
						if((*it).GetRightEyeX() != 0.0 && (*it).GetLeftEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{
							// exceptional case for wrong annotation (//28520_00000.png)
							if((*it).GetFaceX() + (*it).GetFaceWidth() > (*it).GetLeftEyeX())
							{
									input_meta_frames.push_back((*it).GetFrame());
									input_bounding_boxes.push_back(Rect_<double>((*it).GetLeftEyeX(), 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY()), (*it).GetRightEyeX() - (*it).GetLeftEyeX(), (*it).GetNoseBaseY() - 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY())));
									input_image_types.push_back(1);
									input_meta_info.push_back(meta_info);
							}
							else 
							{
								printf("ERROR: Image [%s] ignored because it has wrong annotation!\n", (*it).GetFile().c_str());
							}
						}
						else if((*it).GetRightEyeX() != 0.0 && (*it).GetLeftEyeX() != 0.0)
						{
								input_meta_frames.push_back((*it).GetFrame());
								input_bounding_boxes.push_back(Rect_<double>((*it).GetLeftEyeX(), 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY()), (*it).GetRightEyeX() - (*it).GetLeftEyeX(), ((*it).GetRightEyeX() - (*it).GetLeftEyeX()) * 0.85));
								input_image_types.push_back(7);
								input_meta_info.push_back(meta_info);
						}
						else if((*it).GetLeftEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{
								input_meta_frames.push_back((*it).GetFrame());
								input_bounding_boxes.push_back(Rect_<double>(2 * (*it).GetLeftEyeX() - (*it).GetNoseBaseX(), (*it).GetLeftEyeY(), 2 * ((*it).GetNoseBaseX() - (*it).GetLeftEyeX()), (*it).GetNoseBaseY() - (*it).GetLeftEyeY()));
								input_image_types.push_back(2);
								input_meta_info.push_back(meta_info);
						}
						else if((*it).GetRightEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{
								input_meta_frames.push_back((*it).GetFrame());
								input_bounding_boxes.push_back(Rect_<double>((*it).GetNoseBaseX(), (*it).GetRightEyeY(), 2 * ((*it).GetRightEyeX() - (*it).GetNoseBaseX()), (*it).GetNoseBaseY() - (*it).GetRightEyeY()));
								input_image_types.push_back(3);
								input_meta_info.push_back(meta_info);
						}
						else 
						{
							printf("ERROR: Image [%s] ignored because it doesn't meet contraints!\n", (*it).GetFile().c_str());
					    }
					}
					else
					{
						if((*it).GetRightEyeX() != 0.0 && (*it).GetLeftEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{
								input_meta_frames.push_back((*it).GetFrame());
								input_bounding_boxes.push_back(Rect_<double>((*it).GetLeftEyeX(), 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY()), (*it).GetRightEyeX() - (*it).GetLeftEyeX(), (*it).GetNoseBaseY() - 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY())));
								input_image_types.push_back(4);
								input_meta_info.push_back(meta_info);
						}
						else if((*it).GetRightEyeX() != 0.0 && (*it).GetLeftEyeX() != 0.0)
						{
								input_meta_frames.push_back((*it).GetFrame());
								input_bounding_boxes.push_back(Rect_<double>((*it).GetLeftEyeX(), 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY()), (*it).GetRightEyeX() - (*it).GetLeftEyeX(), ((*it).GetRightEyeX() - (*it).GetLeftEyeX()) * 0.85));
								input_image_types.push_back(8);
								input_meta_info.push_back(meta_info);
						}
						else if((*it).GetLeftEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{
								input_meta_frames.push_back((*it).GetFrame());
								input_bounding_boxes.push_back(Rect_<double>(2 * (*it).GetLeftEyeX() - (*it).GetNoseBaseX(), (*it).GetLeftEyeY(), 2 * ((*it).GetNoseBaseX() - (*it).GetLeftEyeX()), (*it).GetNoseBaseY() - (*it).GetLeftEyeY()));
								input_image_types.push_back(5);
								input_meta_info.push_back(meta_info);
						}
						else if((*it).GetRightEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{		
								input_meta_frames.push_back((*it).GetFrame());
								input_bounding_boxes.push_back(Rect_<double>((*it).GetNoseBaseX(), (*it).GetRightEyeY(), 2 * ((*it).GetRightEyeX() - (*it).GetNoseBaseX()), (*it).GetNoseBaseY() - (*it).GetRightEyeY()));
								input_image_types.push_back(6);
								input_meta_info.push_back(meta_info);
						}
						else 
						{
							printf("ERROR: Image [%s] ignored because it doesn't meet contraints!\n", (*it).GetFile().c_str());
					    }
					}
				}
			}

			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}
	}

	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}
}

void get_image_alg_params(vector<string> &arguments, float &mstep, float &threshSize, float &speed, bool &preOpt, int &numTri){
	
	bool* valid = new bool[arguments.size()];
	
	for(size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-mstep") == 0) 
		{                 
			mstep = atof(arguments[i + 1].c_str());
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if (arguments[i].compare("-thSize") == 0) 
		{                    
			threshSize = atof(arguments[i + 1].c_str());
			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}		
		else if (arguments[i].compare("-speed") == 0) 
		{                    
			speed = atof(arguments[i + 1].c_str());
			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}		
		else if (arguments[i].compare("-preOpt") == 0) 
		{                    
			preOpt = arguments[i + 1].c_str()[0] != '0';
			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}		
		else if (arguments[i].compare("-tri") == 0) 
		{                    
			numTri = atoi(arguments[i + 1].c_str());
			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}
	}

	// Clear up the argument list
	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}
}

void get_image_input_output_params(vector<string> &input_image_files, vector<string> &input_depth_files, vector<string> &output_feature_files, vector<string> &output_image_files,vector<string> &output_model_files, vector<string> &output_lm_files, vector<string> &output_pose_files,
		vector<Rect_<double> > &input_bounding_boxes, bool &bJanus, vector<vector<float> > &input_meta_info, vector<int> &input_image_types, vector<string> &arguments, string &opath, string &olist, vector<string> &inputLMFiles, vector<int> &similarFile, bool &continueWork, string &refDir)
{
	bool* valid = new bool[arguments.size()];
	string lmPath = "";
	string out_pts_dir, out_img_dir;
	vector<int> template_ids;
	vector<int> subject_ids;
	string ipath;
	string meta_file_stem;
	string split;
	string suffix;
	int eye_type;

	// default value
	ipath = "";
	opath = "";
	olist = "";
	eye_type = 0;
	suffix = "";

	for(size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-flm") == 0) 
		{                    
			lmPath = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}	
	}
	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}
	printf("flm %s-\n",lmPath.c_str());
	//main
	for(size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-f") == 0) 
		{                    
			input_image_files.push_back(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if (arguments[i].compare("-fd") == 0) 
		{                    
			input_depth_files.push_back(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}
		else if (arguments[i].compare("-ipath") == 0) 
		{                    
			ipath = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}		
		else if (arguments[i].compare("-opath") == 0) 
		{                    
			opath = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}		
		else if (arguments[i].compare("-eye_type") == 0)
		{
			eye_type = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;
			i++;
		}
		else if (arguments[i].compare("-suffix") == 0)
		{
			suffix = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;
			i++;
		}
		else if (arguments[i].compare("-olist") == 0)
		{
			olist = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;
			i++;
		}
		else if (arguments[i].compare("-con") == 0)
		{
			continueWork = (arguments[i + 1][0] != '0');
			valid[i] = false;
			valid[i+1] = false;
			i++;
		}
		else if (arguments[i].compare("-refDir") == 0)
		{
			refDir = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;
			i++;
		}
		else if (arguments[i].compare("-fmeta") == 0) 
		{
			// parse the -fmeta directory by reading in it
			path meta_file (arguments[i+1]); 

			if (exists(meta_file))   
			{
				meta_file_stem = meta_file.stem().string();
				split = meta_file.parent_path().filename().string();
				std::ifstream in_meta(meta_file.string());
				printf("meta_file_stem split %s--%s--\n",meta_file_stem.c_str(),split.c_str());
				string str;
				JanusMetaData data; 
				vector<JanusMetaData> vdata;
				int cnt = 0;
				while(!in_meta.eof())
				{
					data.Load(in_meta);				
					if (cnt != 0) vdata.push_back(data);
					if (in_meta.eof()) break; 
					cnt++;
				}
				in_meta.close();
				
				if (meta_file_stem.compare(meta_file_stem.size() - 5, 5,"probe") == 0)
				{
					meta_file_stem.replace(meta_file_stem.end() - 6, meta_file_stem.end(),"");
				}
				else if (meta_file_stem.compare(meta_file_stem.size() - 3, 3,"gal") == 0)
				{
					meta_file_stem.replace(meta_file_stem.end() - 4, meta_file_stem.end(),"");
				}
				//if(eye_type == 1) opath = opath + split + "/" + meta_file_stem + "/" + "landmark";
				//else if(eye_type == 2) opath = opath + split + "/" + meta_file_stem + "/" + "landmark";
				//else opath = opath + split + "/" + meta_file_stem + "/"  + "landmark";
				
				printf("opath %s--\n",opath.c_str());
				bJanus = true;
				vector<JanusMetaData>::iterator it; 
				for (it = vdata.begin(); it != vdata.end(); it++) 
				{
					vector<float> meta_info;
					meta_info.push_back((*it).GetFaceX());
					meta_info.push_back((*it).GetFaceY());
					meta_info.push_back((*it).GetFaceWidth());
					meta_info.push_back((*it).GetFaceHeight());
					meta_info.push_back((*it).GetRightEyeX());
					meta_info.push_back((*it).GetRightEyeY());
					meta_info.push_back((*it).GetLeftEyeX());
					meta_info.push_back((*it).GetLeftEyeY());
					meta_info.push_back((*it).GetNoseBaseX());
					meta_info.push_back((*it).GetNoseBaseY());

					if((*it).GetFaceWidth() != 0.0 && (*it).GetFaceHeight() != 0.0)
					{
						if((*it).GetRightEyeX() != 0.0 && (*it).GetLeftEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{
							// exceptional case for wrong annotation (//28520_00000.png)
							if((*it).GetFaceX() + (*it).GetFaceWidth() > (*it).GetLeftEyeX())
							{
								if(eye_type == 0 || eye_type == 2)
								{
									input_image_files.push_back(ipath + (*it).GetFile());

									input_bounding_boxes.push_back(Rect_<double>((*it).GetLeftEyeX(), 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY()), (*it).GetRightEyeX() - (*it).GetLeftEyeX(), (*it).GetNoseBaseY() - 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY())));
									input_image_types.push_back(1);
									input_meta_info.push_back(meta_info);

									template_ids.push_back((*it).GetTemplateId());
									subject_ids.push_back((*it).GetSubjectId());
								}
							}
							else 
							{
								printf("ERROR: Image [%s] ignored because it has wrong annotation!\n", (*it).GetFile().c_str());
							}
						}
						else if((*it).GetRightEyeX() != 0.0 && (*it).GetLeftEyeX() != 0.0)
						{
							if(eye_type == 0 || eye_type == 2)
							{
								input_image_files.push_back(ipath + (*it).GetFile());

								input_bounding_boxes.push_back(Rect_<double>((*it).GetLeftEyeX(), 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY()), (*it).GetRightEyeX() - (*it).GetLeftEyeX(), ((*it).GetRightEyeX() - (*it).GetLeftEyeX()) * 0.85));
								input_image_types.push_back(7);
								input_meta_info.push_back(meta_info);

								template_ids.push_back((*it).GetTemplateId());
								subject_ids.push_back((*it).GetSubjectId());
							}
						}
						else if((*it).GetLeftEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{
							if(eye_type == 0 || eye_type == 1)
							{
								input_image_files.push_back(ipath + (*it).GetFile());

								input_bounding_boxes.push_back(Rect_<double>(2 * (*it).GetLeftEyeX() - (*it).GetNoseBaseX(), (*it).GetLeftEyeY(), 2 * ((*it).GetNoseBaseX() - (*it).GetLeftEyeX()), (*it).GetNoseBaseY() - (*it).GetLeftEyeY()));
								input_image_types.push_back(2);
								input_meta_info.push_back(meta_info);

								template_ids.push_back((*it).GetTemplateId());
								subject_ids.push_back((*it).GetSubjectId());
							}
						}
						else if((*it).GetRightEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{
							if(eye_type == 0 || eye_type == 1)
							{
								input_image_files.push_back(ipath + (*it).GetFile());

								input_bounding_boxes.push_back(Rect_<double>((*it).GetNoseBaseX(), (*it).GetRightEyeY(), 2 * ((*it).GetRightEyeX() - (*it).GetNoseBaseX()), (*it).GetNoseBaseY() - (*it).GetRightEyeY()));
								input_image_types.push_back(3);
								input_meta_info.push_back(meta_info);

								template_ids.push_back((*it).GetTemplateId());
								subject_ids.push_back((*it).GetSubjectId());
							}
						}
						else 
						{
							printf("ERROR: Image [%s] ignored because it doesn't meet contraints!\n", (*it).GetFile().c_str());
					    }
					}
					else
					{
						if((*it).GetRightEyeX() != 0.0 && (*it).GetLeftEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{
							if(eye_type == 0 || eye_type == 2)
							{
								input_image_files.push_back(ipath + (*it).GetFile());

								input_bounding_boxes.push_back(Rect_<double>((*it).GetLeftEyeX(), 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY()), (*it).GetRightEyeX() - (*it).GetLeftEyeX(), (*it).GetNoseBaseY() - 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY())));
								input_image_types.push_back(4);
								input_meta_info.push_back(meta_info);

								template_ids.push_back((*it).GetTemplateId());
								subject_ids.push_back((*it).GetSubjectId());
							}
						}
						else if((*it).GetRightEyeX() != 0.0 && (*it).GetLeftEyeX() != 0.0)
						{
							if(eye_type == 0 || eye_type == 2)
							{
								input_image_files.push_back(ipath + (*it).GetFile());

								input_bounding_boxes.push_back(Rect_<double>((*it).GetLeftEyeX(), 0.5 * ((*it).GetLeftEyeY() + (*it).GetRightEyeY()), (*it).GetRightEyeX() - (*it).GetLeftEyeX(), ((*it).GetRightEyeX() - (*it).GetLeftEyeX()) * 0.85));
								input_image_types.push_back(8);
								input_meta_info.push_back(meta_info);

								template_ids.push_back((*it).GetTemplateId());
								subject_ids.push_back((*it).GetSubjectId());
							}
						}
						else if((*it).GetLeftEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{
							if(eye_type == 0 || eye_type == 1)
							{
								input_image_files.push_back(ipath + (*it).GetFile());

								input_bounding_boxes.push_back(Rect_<double>(2 * (*it).GetLeftEyeX() - (*it).GetNoseBaseX(), (*it).GetLeftEyeY(), 2 * ((*it).GetNoseBaseX() - (*it).GetLeftEyeX()), (*it).GetNoseBaseY() - (*it).GetLeftEyeY()));
								input_image_types.push_back(5);
								input_meta_info.push_back(meta_info);

								template_ids.push_back((*it).GetTemplateId());
								subject_ids.push_back((*it).GetSubjectId());
							}
						}
						else if((*it).GetRightEyeX() != 0.0 && (*it).GetNoseBaseX() != 0.0)
						{		
							if(eye_type == 0 || eye_type == 1)
							{
								input_image_files.push_back(ipath + (*it).GetFile());

								input_bounding_boxes.push_back(Rect_<double>((*it).GetNoseBaseX(), (*it).GetRightEyeY(), 2 * ((*it).GetRightEyeX() - (*it).GetNoseBaseX()), (*it).GetNoseBaseY() - (*it).GetRightEyeY()));
								input_image_types.push_back(6);
								input_meta_info.push_back(meta_info);

								template_ids.push_back((*it).GetTemplateId());
								subject_ids.push_back((*it).GetSubjectId());
							}
						}
						else 
						{
							printf("ERROR: Image [%s] ignored because it doesn't meet contraints!\n", (*it).GetFile().c_str());
					    }
					}
				}
			}

			for(size_t i=0; i < input_image_files.size(); ++i)
			{
				path image_loc(input_image_files[i]);

				path fname = image_loc.filename();
				fname = fname.replace_extension("jpg");

				stringstream tstr, sstr;
				tstr << template_ids[i];
				sstr << subject_ids[i];
				string str_tempalte = tstr.str();
				string str_subject = sstr.str();
				
				output_image_files.push_back(opath + "/" + fname.stem().string() /* + "_T" + str_tempalte */ + "_S" + str_subject + "_" + suffix + ".jpg");
			}

			for(size_t i=0; i < input_image_files.size(); ++i)
			{
				path image_loc(input_image_files[i]);
				
				path fname = image_loc.filename();
				fname = fname.replace_extension("pts");

				stringstream tstr, sstr;
				tstr << template_ids[i];
				sstr << subject_ids[i];
				string str_tempalte = tstr.str();
				string str_subject = sstr.str();
				
				output_feature_files.push_back(opath + "/" + fname.stem().string() /* + "_T" + str_tempalte */ + "_S" + str_subject + "_" + suffix + ".pts");
				if (lmPath.length() > 0){
					inputLMFiles.push_back(lmPath + "/" + fname.stem().string() /* + "_T" + str_tempalte */+ "_S" + str_subject + "_" + suffix + ".pts");
				//printf("f %s-\n",inputLMFiles[inputLMFiles.size()-1].c_str());
}

			}
			
			for(size_t i=0; i < input_image_files.size(); ++i)
			{
				path image_loc(input_image_files[i]);
				
				path fname = image_loc.filename();
				fname = fname.replace_extension("ply");

				stringstream tstr, sstr;
				tstr << template_ids[i];
				sstr << subject_ids[i];
				string str_tempalte = tstr.str();
				string str_subject = sstr.str();
				
				output_model_files.push_back(opath + "/" + fname.stem().string() /*+ "_T" + str_tempalte*/ + "_S" + str_subject + ".ply");
				output_lm_files.push_back(opath + "/" + fname.stem().string()/* + "_T" + str_tempalte*/ + "_S" + str_subject + "_lm.ply");
				output_pose_files.push_back(opath + "/" + fname.stem().string() /*+ "_T" + str_tempalte*/ + "_S" + str_subject + ".pose");
			}
			similarFile.clear();
			similarFile.push_back(-1);
			//FILE* fsim = fopen("similar.txt","w");
			//fprintf(fsim,"-1\n");
			for(size_t i=1; i < input_image_files.size(); ++i)
			{
				bool found = false;
				for (size_t j=0; j < i; ++j){
					if (subject_ids[i] == subject_ids[j] && input_image_files[i] == input_image_files[j]){
						similarFile.push_back(j);
						//fprintf(fsim,"%d\n",j);
						found = true;
						break;
					}
				}
				if (!found) {
					similarFile.push_back(-1);
					//fprintf(fsim,"%d\n",-1);
				}
			}
			//fclose(fsim);
			//getchar();

			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}
		else if (arguments[i].compare("-fdir") == 0) 
		{                    

			// parse the -fdir directory by reading in all of the .png and .jpg files in it
			path image_directory (arguments[i+1]); 

			try
			{
				 // does the file exist and is it a directory
				if (exists(image_directory) && is_directory(image_directory))   
				{

					vector<path> file_in_directory;                                
					copy(directory_iterator(image_directory), directory_iterator(), back_inserter(file_in_directory));

					for (vector<path>::const_iterator file_iterator (file_in_directory.begin()); file_iterator != file_in_directory.end(); ++file_iterator)
					{
						// Possible image extension .jpg and .png
						if(file_iterator->extension().string().compare(".jpg") == 0 || file_iterator->extension().string().compare(".png") == 0)
						{
								
								
							input_image_files.push_back(file_iterator->string());
								
							// If there exists a .txt file corresponding to the image, it is assumed that it contains a bounding box definition for a face
							// [minx, miny, maxx, maxy]
							path current_file = *file_iterator;
							path bbox = current_file.replace_extension("txt");

							// If there is a bounding box file push it to the list of bounding boxes
							if(exists(bbox))
							{

								std::ifstream in_bbox(bbox.string());

								double min_x, min_y, max_x, max_y;

								in_bbox >> min_x >> min_y >> max_x >> max_y;

								in_bbox.close();

								input_bounding_boxes.push_back(Rect_<double>(min_x, min_y, max_x - min_x, max_y - min_y));
							}
						}
					}
				}
			}
			catch (const filesystem_error& ex)
			{
				cout << ex.what() << '\n';
			}

			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}
		else if (arguments[i].compare("-flist") == 0) 
		{                    
			std::ifstream in_meta(arguments[i+1]);
			int cnt = 0;
			string line;
			while(!in_meta.eof())
			{
				getline(in_meta, line);
				if (line.length() > 4) {
					istringstream ins(line);
					input_image_files.push_back(line);
				}	
			}
			in_meta.close();

			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}
		else if (arguments[i].compare("-ofdir") == 0) 
		{
			out_pts_dir = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;
			i++;
		}
		else if (arguments[i].compare("-oidir") == 0) 
		{
			out_img_dir = arguments[i + 1];
			valid[i] = false;
			valid[i+1] = false;
			i++;
		}
		else if (arguments[i].compare("-of") == 0)
		{
			output_model_files.push_back(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;
			i++;
		} 
		else if (arguments[i].compare("-oi") == 0)
		{
			output_image_files.push_back(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;
			i++;
		} 
		else if (arguments[i].compare("-help") == 0)
		{
			cout << "Input output files are defined as: -f <infile (can have multiple ones)> -of <where detected landmarks should be stored(can have multiple ones)> -oi <where should images with detected landmarks should be written (can have multiple ones)> -fdir <the directory containing .png and .jpg files to be processed (with optional .txt files corresponding to EACH image containing the bounding boxes> " << endl; // Inform the user of how to use the program				
		}
	}

	// If any output directories are defined populate them based on image names
	if(!out_img_dir.empty())
	{
		for(size_t i=0; i < input_image_files.size(); ++i)
		{
			path image_loc(input_image_files[i]);

			path fname = image_loc.filename();
			fname = fname.replace_extension("jpg");
			output_image_files.push_back(out_img_dir + "/" + fname.string());
		}
	}

	if(!out_pts_dir.empty())
	{
		for(size_t i=0; i < input_image_files.size(); ++i)
		{
			path image_loc(input_image_files[i]);

			path fname = image_loc.filename();
			fname = fname.replace_extension("ply");
			output_model_files.push_back(out_pts_dir + "/" + fname.string());
			output_feature_files.push_back(out_pts_dir + "/" + fname.stem().string() + ".pts");
			output_lm_files.push_back(out_pts_dir + "/" + fname.stem().string() + "_lm.ply");
			output_pose_files.push_back(out_pts_dir + "/" + fname.stem().string() + ".pose");
			if (lmPath.length() > 0){
				inputLMFiles.push_back(lmPath + "/" + fname.stem().string() + ".pts");
				printf("f %s-\n",inputLMFiles[inputLMFiles.size()-1].c_str());
			}
		}
	}

	// Make sure the same number of images and bounding boxes is present, if any bounding boxes are defined
	if(input_bounding_boxes.size() > 0)
	{
		assert(input_bounding_boxes.size() == input_image_files.size());
	}

	// Clear up the argument list
	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}

}


//===========================================================================
// Fast patch expert response computation (linear model across a ROI) using normalised cross-correlation
//===========================================================================

// A helper for matchTemplate
void crossCorr_m( const cv::Mat& img, cv::Mat& img_dft, const cv::Mat& _templ, map<int, cv::Mat>& _templ_dfts, cv::Mat& corr,
                cv::Size corrsize, int ctype,
                cv::Point anchor, double delta, int borderType )
{
    const double blockScale = 4.5;
    const int minBlockSize = 256;
    std::vector<uchar> buf;

    cv::Mat templ = _templ;
    int depth = img.depth(), cn = img.channels();
    int tdepth = templ.depth(), tcn = templ.channels();
    int cdepth = CV_MAT_DEPTH(ctype), ccn = CV_MAT_CN(ctype);

    CV_Assert( img.dims <= 2 && templ.dims <= 2 && corr.dims <= 2 );

    if( depth != tdepth && tdepth != std::max(CV_32F, depth) )
    {
        _templ.convertTo(templ, std::max(CV_32F, depth));
        tdepth = templ.depth();
    }

	// make sure both of their depths are the same?
    CV_Assert( depth == tdepth || tdepth == CV_32F);
    CV_Assert( corrsize.height <= img.rows + templ.rows - 1 &&
               corrsize.width <= img.cols + templ.cols - 1 );

    CV_Assert( ccn == 1 || delta == 0 );

    corr.create(corrsize, ctype);

    int maxDepth = depth > CV_8S ? CV_64F : std::max(std::max(CV_32F, tdepth), cdepth);
    cv::Size blocksize, dftsize;

    blocksize.width = cvRound(templ.cols*blockScale);
    blocksize.width = std::max( blocksize.width, minBlockSize - templ.cols + 1 );
    blocksize.width = std::min( blocksize.width, corr.cols );
    blocksize.height = cvRound(templ.rows*blockScale);
    blocksize.height = std::max( blocksize.height, minBlockSize - templ.rows + 1 );
    blocksize.height = std::min( blocksize.height, corr.rows );

    dftsize.width = std::max(cv::getOptimalDFTSize(blocksize.width + templ.cols - 1), 2);
    dftsize.height = cv::getOptimalDFTSize(blocksize.height + templ.rows - 1);
    if( dftsize.width <= 0 || dftsize.height <= 0 )
        CV_Error( CV_StsOutOfRange, "the input arrays are too big" );

    // recompute block size
    blocksize.width = dftsize.width - templ.cols + 1;
    blocksize.width = MIN( blocksize.width, corr.cols );
    blocksize.height = dftsize.height - templ.rows + 1;
    blocksize.height = MIN( blocksize.height, corr.rows );

    cv::Mat dftImg( dftsize, maxDepth );

    int i, k, bufSize = 0;
    if( tcn > 1 && tdepth != maxDepth )
        bufSize = templ.cols*templ.rows*CV_ELEM_SIZE(tdepth);

    if( cn > 1 && depth != maxDepth )
        bufSize = std::max( bufSize, (blocksize.width + templ.cols - 1)*
            (blocksize.height + templ.rows - 1)*CV_ELEM_SIZE(depth));

    if( (ccn > 1 || cn > 1) && cdepth != maxDepth )
        bufSize = std::max( bufSize, blocksize.width*blocksize.height*CV_ELEM_SIZE(cdepth));

    buf.resize(bufSize);

	cv::Mat dftTempl( dftsize.height*tcn, dftsize.width, maxDepth );

	// if this has not been precomputer, precompute it, otherwise use it
	if(_templ_dfts.find(dftsize.width) == _templ_dfts.end())
	{

		// compute DFT of each template plane
		for( k = 0; k < tcn; k++ )
		{
			int yofs = k*dftsize.height;
			cv::Mat src = templ;
			cv::Mat dst(dftTempl, cv::Rect(0, yofs, dftsize.width, dftsize.height));
			cv::Mat dst1(dftTempl, cv::Rect(0, yofs, templ.cols, templ.rows));

			if( tcn > 1 )
			{
				src = tdepth == maxDepth ? dst1 : cv::Mat(templ.size(), tdepth, &buf[0]);
				int pairs[] = {k, 0};
				mixChannels(&templ, 1, &src, 1, pairs, 1);
			}

			if( dst1.data != src.data )
				src.convertTo(dst1, dst1.depth());

			if( dst.cols > templ.cols )
			{
				cv::Mat part(dst, cv::Range(0, templ.rows), cv::Range(templ.cols, dst.cols));
				part = cv::Scalar::all(0);
			}
			dft(dst, dst, 0, templ.rows);
		}
		_templ_dfts[dftsize.width] = dftTempl;
	}
	else
	{
		// use the precomputed version
		dftTempl = _templ_dfts.find(dftsize.width)->second;
	}

    int tileCountX = (corr.cols + blocksize.width - 1)/blocksize.width;
    int tileCountY = (corr.rows + blocksize.height - 1)/blocksize.height;
    int tileCount = tileCountX * tileCountY;

    cv::Size wholeSize = img.size();
    cv::Point roiofs(0,0);
    cv::Mat img0 = img;

    if( !(borderType & cv::BORDER_ISOLATED) )
    {
        img.locateROI(wholeSize, roiofs);
        img0.adjustROI(roiofs.y, wholeSize.height-img.rows-roiofs.y,
                       roiofs.x, wholeSize.width-img.cols-roiofs.x);
    }
    borderType |= cv::BORDER_ISOLATED;

    // calculate correlation by blocks
    for( i = 0; i < tileCount; i++ )
    {

        int x = (i%tileCountX)*blocksize.width;
        int y = (i/tileCountX)*blocksize.height;

        cv::Size bsz(std::min(blocksize.width, corr.cols - x),
                 std::min(blocksize.height, corr.rows - y));
        cv::Size dsz(bsz.width + templ.cols - 1, bsz.height + templ.rows - 1);
        int x0 = x - anchor.x + roiofs.x, y0 = y - anchor.y + roiofs.y;
        int x1 = std::max(0, x0), y1 = std::max(0, y0);
        int x2 = std::min(img0.cols, x0 + dsz.width);
        int y2 = std::min(img0.rows, y0 + dsz.height);
        cv::Mat src0(img0, cv::Range(y1, y2), cv::Range(x1, x2));
        cv::Mat dst(dftImg, cv::Rect(0, 0, dsz.width, dsz.height));
        cv::Mat dst1(dftImg, cv::Rect(x1-x0, y1-y0, x2-x1, y2-y1));
        cv::Mat cdst(corr, cv::Rect(x, y, bsz.width, bsz.height));

        for( k = 0; k < cn; k++ )
        {
            cv::Mat src = src0;
            dftImg = cv::Scalar::all(0);

            if( cn > 1 )
            {
                src = depth == maxDepth ? dst1 : cv::Mat(y2-y1, x2-x1, depth, &buf[0]);
                int pairs[] = {k, 0};
                mixChannels(&src0, 1, &src, 1, pairs, 1);
            }

            if( dst1.data != src.data )
                src.convertTo(dst1, dst1.depth());

            if( x2 - x1 < dsz.width || y2 - y1 < dsz.height )
                copyMakeBorder(dst1, dst, y1-y0, dst.rows-dst1.rows-(y1-y0),
                               x1-x0, dst.cols-dst1.cols-(x1-x0), borderType);
			if(img_dft.empty())
			{
				dft( dftImg, dftImg, 0, dsz.height );
				img_dft = dftImg.clone();
			}
			else
			{
				dftImg = img_dft.clone();
			}
			cv::Mat dftTempl1(dftTempl, cv::Rect(0, tcn > 1 ? k*dftsize.height : 0,
                                         dftsize.width, dftsize.height));
            mulSpectrums(dftImg, dftTempl1, dftImg, 0, true);
            dft( dftImg, dftImg, cv::DFT_INVERSE + cv::DFT_SCALE, bsz.height );

            src = dftImg(cv::Rect(0, 0, bsz.width, bsz.height));

            if( ccn > 1 )
            {
                if( cdepth != maxDepth )
                {
                    cv::Mat plane(bsz, cdepth, &buf[0]);
                    src.convertTo(plane, cdepth, 1, delta);
                    src = plane;
                }
                int pairs[] = {0, k};
                mixChannels(&src, 1, &cdst, 1, pairs, 1);
            }
            else
            {
                if( k == 0 )
                    src.convertTo(cdst, cdepth, 1, delta);
                else
                {
                    if( maxDepth != cdepth )
                    {
                        cv::Mat plane(bsz, cdepth, &buf[0]);
                        src.convertTo(plane, cdepth);
                        src = plane;
                    }
                    add(src, cdst, cdst);
                }
            }
        }
    }
}

/*****************************************************************************************/
// The template matching code from OpenCV with some precomputation optimisations
void matchTemplate_m( cv::InputArray _img, cv::Mat& _img_dft, cv::Mat& _integral_img, cv::Mat& _integral_img_sq, cv::InputArray _templ, map<int, cv::Mat>& _templ_dfts, cv::Mat_<float>& _result, int method )
{
    CV_Assert( CV_TM_SQDIFF <= method && method <= CV_TM_CCOEFF_NORMED );

    int numType = method == CV_TM_CCORR || method == CV_TM_CCORR_NORMED ? 0 :
                  method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED ? 1 : 2;
    bool isNormed = method == CV_TM_CCORR_NORMED ||
                    method == CV_TM_SQDIFF_NORMED ||
                    method == CV_TM_CCOEFF_NORMED;

    cv::Mat img = _img.getMat(), templ = _templ.getMat();
    if( img.rows < templ.rows || img.cols < templ.cols )
        std::swap(img, templ);

    CV_Assert( (img.depth() == CV_8U || img.depth() == CV_32F) &&
               img.type() == templ.type() );

    cv::Size corrSize(img.cols - templ.cols + 1, img.rows - templ.rows + 1);
    _result.create(corrSize);

    cv::Mat result = _result;

    int cn = img.channels();
    crossCorr_m( img, _img_dft, templ, _templ_dfts, result, result.size(), result.type(), cv::Point(0,0), 0, 0);

    if( method == CV_TM_CCORR )
        return;

    double invArea = 1./((double)templ.rows * templ.cols);

    cv::Mat sum, sqsum;
    cv::Scalar templMean, templSdv;
    double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
    double templNorm = 0, templSum2 = 0;

    if( method == CV_TM_CCOEFF )
    {
		if(_integral_img.empty())
		{
			integral(img, _integral_img, CV_64F);
		}
		
		sum = _integral_img;
        templMean = mean(templ);
    }
    else
    {
		if(_integral_img.empty())
		{
			integral(img, _integral_img, _integral_img_sq, CV_64F);			
		}
		sum = _integral_img;
		sqsum = _integral_img_sq;

        meanStdDev( templ, templMean, templSdv );

        templNorm = CV_SQR(templSdv[0]) + CV_SQR(templSdv[1]) +
                    CV_SQR(templSdv[2]) + CV_SQR(templSdv[3]);

        if( templNorm < DBL_EPSILON && method == CV_TM_CCOEFF_NORMED )
        {
            result = cv::Scalar::all(1);
            return;
        }

        templSum2 = templNorm +
                     CV_SQR(templMean[0]) + CV_SQR(templMean[1]) +
                     CV_SQR(templMean[2]) + CV_SQR(templMean[3]);

        if( numType != 1 )
        {
            templMean = cv::Scalar::all(0);
            templNorm = templSum2;
        }

        templSum2 /= invArea;
        templNorm = sqrt(templNorm);
        templNorm /= sqrt(invArea); // care of accuracy here

        q0 = (double*)sqsum.data;
        q1 = q0 + templ.cols*cn;
        q2 = (double*)(sqsum.data + templ.rows*sqsum.step);
        q3 = q2 + templ.cols*cn;
    }

    double* p0 = (double*)sum.data;
    double* p1 = p0 + templ.cols*cn;
    double* p2 = (double*)(sum.data + templ.rows*sum.step);
    double* p3 = p2 + templ.cols*cn;

    int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
    int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

    int i, j, k;

    for( i = 0; i < result.rows; i++ )
    {
        float* rrow = (float*)(result.data + i*result.step);
        int idx = i * sumstep;
        int idx2 = i * sqstep;

        for( j = 0; j < result.cols; j++, idx += cn, idx2 += cn )
        {
            double num = rrow[j], t;
            double wndMean2 = 0, wndSum2 = 0;

            if( numType == 1 )
            {
                for( k = 0; k < cn; k++ )
                {
                    t = p0[idx+k] - p1[idx+k] - p2[idx+k] + p3[idx+k];
                    wndMean2 += CV_SQR(t);
                    num -= t*templMean[k];
                }

                wndMean2 *= invArea;
            }

            if( isNormed || numType == 2 )
            {
                for( k = 0; k < cn; k++ )
                {
                    t = q0[idx2+k] - q1[idx2+k] - q2[idx2+k] + q3[idx2+k];
                    wndSum2 += t;
                }

                if( numType == 2 )
                {
                    num = wndSum2 - 2*num + templSum2;
                    num = MAX(num, 0.);
                }
            }

            if( isNormed )
            {
                t = sqrt(MAX(wndSum2 - wndMean2,0))*templNorm;
                if( fabs(num) < t )
                    num /= t;
                else if( fabs(num) < t*1.125 )
                    num = num > 0 ? 1 : -1;
                else
                    num = method != CV_TM_SQDIFF_NORMED ? 0 : 1;
            }

            rrow[j] = (float)num;
        }
    }
}

//===========================================================================
// Point set and landmark manipulation functions
//===========================================================================
// Using Kabsch's algorithm for aligning shapes
//This assumes that align_from and align_to are already mean normalised
Matx22d AlignShapesKabsch2D(const Mat_<double>& align_from, const Mat_<double>& align_to )
{

	cv::SVD svd(align_from.t() * align_to);
    
	// make sure no reflection is there
	// corr ensures that we do only rotaitons and not reflections
	double d = cv::determinant(svd.vt.t() * svd.u.t());

	cv::Matx22d corr = cv::Matx22d::eye();
	if(d > 0)
	{
		corr(1,1) = 1;
	}
	else
	{
		corr(1,1) = -1;
	}

    Matx22d R;
	Mat(svd.vt.t()*cv::Mat(corr)*svd.u.t()).copyTo(R);

	return R;
}

//=============================================================================
// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
Matx22d AlignShapesWithScale(cv::Mat_<double>& src, cv::Mat_<double> dst)
{
	int n = src.rows;

	// First we mean normalise both src and dst
	double mean_src_x = cv::mean(src.col(0))[0];
	double mean_src_y = cv::mean(src.col(1))[0];

	double mean_dst_x = cv::mean(dst.col(0))[0];
	double mean_dst_y = cv::mean(dst.col(1))[0];

	Mat_<double> src_mean_normed = src.clone();
	src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
	src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

	Mat_<double> dst_mean_normed = dst.clone();
	dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
	dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;

	// Find the scaling factor of each
	Mat src_sq;
	cv::pow(src_mean_normed, 2, src_sq);

	Mat dst_sq;
	cv::pow(dst_mean_normed, 2, dst_sq);

	double s_src = sqrt(cv::sum(src_sq)[0]/n);
	double s_dst = sqrt(cv::sum(dst_sq)[0]/n);

	src_mean_normed = src_mean_normed / s_src;
	dst_mean_normed = dst_mean_normed / s_dst;

	double s = s_dst / s_src;

	// Get the rotation
	Matx22d R = AlignShapesKabsch2D(src_mean_normed, dst_mean_normed);
		
	Matx22d	A;
	Mat(s * R).copyTo(A);

	Mat_<double> aligned = (Mat(Mat(A) * src.t())).t();
    Mat_<double> offset = dst - aligned;

	double t_x =  cv::mean(offset.col(0))[0];
	double t_y =  cv::mean(offset.col(1))[0];
    
	return A;

}


//===========================================================================
// Visualisation functions
//===========================================================================
void Project(Mat_<double>& dest, const Mat_<double>& mesh, Size size, double fx, double fy, double cx, double cy)
{
	dest = Mat_<double>(mesh.rows,2, 0.0);

	int num_points = mesh.rows;

	double X, Y, Z;


	Mat_<double>::const_iterator mData = mesh.begin();
	Mat_<double>::iterator projected = dest.begin();

	for(int i = 0;i < num_points; i++)
	{
		// Get the points
		X = *(mData++);
		Y = *(mData++);
		Z = *(mData++);
			
		double x;
		double y;

		// if depth is 0 the projection is different
		if(Z != 0)
		{
			x = ((X * fx / Z) + cx);
			y = ((Y * fy / Z) + cy);
		}
		else
		{
			x = X;
			y = Y;
		}

		// Clamping to image size
		if( x < 0 )	
		{
			x = 0.0;
		}
		else if (x > size.width - 1)
		{
			x = size.width - 1.0f;
		}
		if( y < 0 )
		{
			y = 0.0;
		}
		else if( y > size.height - 1) 
		{
			y = size.height - 1.0f;
		}

		// Project and store in dest matrix
		(*projected++) = x;
		(*projected++) = y;
	}

}

void DrawBox(Mat image, Vec6d pose, Scalar color, int thickness, float fx, float fy, float cx, float cy)
{
	double boxVerts[] = {-1, 1, -1,
						1, 1, -1,
						1, 1, 1,
						-1, 1, 1,
						1, -1, 1,
						1, -1, -1,
						-1, -1, -1,
						-1, -1, 1};
	Mat_<double> box = Mat(8, 3, CV_64F, boxVerts).clone() * 100;


	Matx33d rot = CLMTracker::Euler2RotationMatrix(Vec3d(pose[3], pose[4], pose[5]));
	Mat_<double> rotBox;
	
	// Rotate the box
	rotBox = Mat(rot) * box.t();
	rotBox = rotBox.t();

	rotBox.col(0) = rotBox.col(0) + pose[0];
	rotBox.col(1) = rotBox.col(1) + pose[1];
	rotBox.col(2) = rotBox.col(2) + pose[2];

	// draw the lines
	Mat_<double> rotBoxProj;
	Project(rotBoxProj, rotBox, image.size(), fx, fy, cx, cy);
	
	Mat_<double> begin;
	Mat_<double> end;

	rotBoxProj.row(0).copyTo(begin);
	rotBoxProj.row(1).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
		
	rotBoxProj.row(1).copyTo(begin);
	rotBoxProj.row(2).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
	
	rotBoxProj.row(2).copyTo(begin);
	rotBoxProj.row(3).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
	
	rotBoxProj.row(0).copyTo(begin);
	rotBoxProj.row(3).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
	
	rotBoxProj.row(2).copyTo(begin);
	rotBoxProj.row(4).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
	
	rotBoxProj.row(1).copyTo(begin);
	rotBoxProj.row(5).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
	
	rotBoxProj.row(0).copyTo(begin);
	rotBoxProj.row(6).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
	
	rotBoxProj.row(3).copyTo(begin);
	rotBoxProj.row(7).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
	
	rotBoxProj.row(6).copyTo(begin);
	rotBoxProj.row(5).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
	
	rotBoxProj.row(5).copyTo(begin);
	rotBoxProj.row(4).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
		
	rotBoxProj.row(4).copyTo(begin);
	rotBoxProj.row(7).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
	
	rotBoxProj.row(7).copyTo(begin);
	rotBoxProj.row(6).copyTo(end);
	cv::line(image, Point((int)begin.at<double>(0), (int)begin.at<double>(1)), Point((int)end.at<double>(0), (int)end.at<double>(1)), color, thickness);
	

}

void Draw(cv::Mat img, const Mat& shape2D, Mat& visibilities)
{
	int n = shape2D.rows/2;

	for( int i = 0; i < n; ++i)
	{		
		if(visibilities.at<int>(i))
		{
			Point featurePoint((int)shape2D.at<double>(i), (int)shape2D.at<double>(i +n));

			// A rough heuristic for drawn point size
			int thickness = (int)std::ceil(5.0* ((double)img.cols) / 640.0);
			int thickness_2 = (int)std::ceil(1.5* ((double)img.cols) / 640.0);

			cv::circle(img, featurePoint, 1, Scalar(0,0,255), 2/*thickness*/);
			cv::circle(img, featurePoint, 1, Scalar(255,0,0), 1/*thickness_2*/);
		}
	}
	
}

void Draw(cv::Mat img, CLM& clm_model)
{

	int idx = clm_model.patch_experts.GetViewIdx(clm_model.params_global, 0);

	// Because we only draw visible points, need to find which points patch experts consider visible at a certain orientation
	Draw(img, clm_model.detected_landmarks, clm_model.patch_experts.visibilities[0][idx]);

}

//===========================================================================
// Angle representation conversion helpers
//===========================================================================

// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
Matx33d Euler2RotationMatrix(const Vec3d& eulerAngles)
{
	Matx33d rotation_matrix;

	double s1 = sin(eulerAngles[0]);
	double s2 = sin(eulerAngles[1]);
	double s3 = sin(eulerAngles[2]);

	double c1 = cos(eulerAngles[0]);
	double c2 = cos(eulerAngles[1]);
	double c3 = cos(eulerAngles[2]);

	rotation_matrix(0,0) = c2 * c3;
	rotation_matrix(0,1) = -c2 *s3;
	rotation_matrix(0,2) = s2;
	rotation_matrix(1,0) = c1 * s3 + c3 * s1 * s2;
	rotation_matrix(1,1) = c1 * c3 - s1 * s2 * s3;
	rotation_matrix(1,2) = -c2 * s1;
	rotation_matrix(2,0) = s1 * s3 - c1 * c3 * s2;
	rotation_matrix(2,1) = c3 * s1 + c1 * s2 * s3;
	rotation_matrix(2,2) = c1 * c2;

	return rotation_matrix;
}

// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
Vec3d RotationMatrix2Euler(const Matx33d& rotation_matrix)
{
	double q0 = sqrt( 1 + rotation_matrix(0,0) + rotation_matrix(1,1) + rotation_matrix(2,2) ) / 2.0;
	double q1 = (rotation_matrix(2,1) - rotation_matrix(1,2)) / (4.0*q0) ;
	double q2 = (rotation_matrix(0,2) - rotation_matrix(2,0)) / (4.0*q0) ;
	double q3 = (rotation_matrix(1,0) - rotation_matrix(0,1)) / (4.0*q0) ;

	double t1 = 2.0 * (q0*q2 + q1*q3);

	double yaw  = asin(2.0 * (q0*q2 + q1*q3));
	double pitch= atan2(2.0 * (q0*q1-q2*q3), q0*q0-q1*q1-q2*q2+q3*q3); 
	double roll = atan2(2.0 * (q0*q3-q1*q2), q0*q0+q1*q1-q2*q2-q3*q3);
    
	return Vec3d(pitch, yaw, roll);
}

Vec3d Euler2AxisAngle(const Vec3d& euler)
{
	Matx33d rotMatrix = CLMTracker::Euler2RotationMatrix(euler);
	Vec3d axis_angle;
	cv::Rodrigues(rotMatrix, axis_angle);
	return axis_angle;
}

Vec3d AxisAngle2Euler(const Vec3d& axis_angle)
{
	Matx33d rotation_matrix;
	cv::Rodrigues(axis_angle, rotation_matrix);
	return RotationMatrix2Euler(rotation_matrix);
}

Matx33d AxisAngle2RotationMatrix(const Vec3d& axis_angle)
{
	Matx33d rotation_matrix;
	cv::Rodrigues(axis_angle, rotation_matrix);
	return rotation_matrix;
}

Vec3d RotationMatrix2AxisAngle(const Matx33d& rotation_matrix)
{
	Vec3d axis_angle;
	cv::Rodrigues(rotation_matrix, axis_angle);
	return axis_angle;
}

//===========================================================================

//============================================================================
// Face detection helpers
//============================================================================
bool DetectFaces(vector<Rect_<double> >& o_regions, const Mat_<uchar>& intensity)
{
	CascadeClassifier classifier("./classifiers/haarcascade_frontalface_alt.xml");
	if(classifier.empty())
	{
		cout << "Couldn't load the Haar cascade classifier" << endl;
		return false;
	}
	else
	{
		return DetectFaces(o_regions, intensity, classifier);
	}

}

bool DetectFaces(vector<Rect_<double> >& o_regions, const Mat_<uchar>& intensity, CascadeClassifier& classifier)
{
		
	vector<Rect> face_detections;
	classifier.detectMultiScale(intensity, face_detections, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING, Size(50, 50)); 		

	// Convert from int bounding box do a double one with corrections
	o_regions.resize(face_detections.size());

	for( size_t face = 0; face < o_regions.size(); ++face)
	{
		// OpenCV is overgenerous with face size and y location is off
		// CLM expect the bounding box to encompass from eyebrow to chin in y, and from cheeck outline to cheeck outline in x, so we need to compensate

		// Correct for scale
		o_regions[face].width = face_detections[face].width * 0.88; 
		o_regions[face].height = face_detections[face].height * 0.88;

		o_regions[face].y = face_detections[face].x + (face_detections[face].height - o_regions[face].height)/2;		
		o_regions[face].x = face_detections[face].x + (face_detections[face].width - o_regions[face].width)/2;
		
		// Shift face down
		o_regions[face].y = face_detections[face].y + face_detections[face].height * 0.13;
		
		
	}
	return o_regions.size() > 0;
}

bool DetectSingleFace(Rect_<double>& o_region, const Mat_<uchar>& intensity_image, CascadeClassifier& classifier)
{
	// The tracker can return multiple faces
	vector<Rect_<double> > face_detections;
				
	bool detect_success = CLMTracker::DetectFaces(face_detections, intensity_image, classifier);
					
	if(detect_success)
	{
		
		if(face_detections.size() > 1)
		{
			// keep the closest one (this is a hack for the experiment)
			double best = -1;
			int bestIndex = -1;
			for( size_t i = 0; i < face_detections.size(); ++i)
			{
				// Pick a closest face
				if(i == 0 || face_detections[i].width > best)
				{
					bestIndex = i;
					best = face_detections[i].width;
				}									
			}

			o_region = face_detections[bestIndex];

		}
		else
		{	
			o_region = face_detections[0];
		}				
	
	}
	else
	{
		// if not detected
		o_region = Rect_<double>(0,0,0,0);
	}
	return detect_success;
}

//============================================================================
// Matrix reading functionality
//============================================================================

// Reading in a matrix from a stream
void ReadMat(std::ifstream& stream, Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row,col,type;
	
	stream >> row >> col >> type;

	output_mat = cv::Mat(row, col, type);
		
	switch(output_mat.type())
	{
		case CV_64FC1: 
		{
			cv::MatIterator_<double> begin_it = output_mat.begin<double>();
			cv::MatIterator_<double> end_it = output_mat.end<double>();
			
			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_32FC1:
		{
			cv::MatIterator_<float> begin_it = output_mat.begin<float>();
			cv::MatIterator_<float> end_it = output_mat.end<float>();

			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_32SC1:
		{
			cv::MatIterator_<int> begin_it = output_mat.begin<int>();
			cv::MatIterator_<int> end_it = output_mat.end<int>();
			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_8UC1:
		{
			cv::MatIterator_<uchar> begin_it = output_mat.begin<uchar>();
			cv::MatIterator_<uchar> end_it = output_mat.end<uchar>();
			while(begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		default:
			printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", __FILE__,__LINE__,output_mat.type()); abort();

	}
}

void ReadMatBin(std::ifstream& stream, Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row, col, type;
	
	stream.read ((char*)&row, 4);
	stream.read ((char*)&col, 4);
	stream.read ((char*)&type, 4);
	
	output_mat = cv::Mat(row, col, type);
	int size = output_mat.rows * output_mat.cols * output_mat.elemSize();
	stream.read((char *)output_mat.data, size);

}

// Skipping lines that start with # (together with empty lines)
void SkipComments(std::ifstream& stream)
{	
	while(stream.peek() == '#' || stream.peek() == '\n'|| stream.peek() == ' ' || stream.peek() == '\r')
	{
		std::string skipped;
		std::getline(stream, skipped);
	}
}

}
