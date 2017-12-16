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

#include "CLM.h"
#include "CLMTracker.h"
#include "CLMParameters.h"
#include "CLM_utils.h"

#include <fstream>
#include <sstream>
//#include "cv.h"
//#include "highgui.h"
#include <opencv2/opencv.hpp>

#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include "FaceServices2.h"

using namespace std;
using namespace cv;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 1; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

bool checkFile(string fname, int nline)
{
        if (!boost::filesystem::exists(fname)) return false;
	int count = 0;
	char text[200];
	FILE* file = fopen(fname.c_str(),"r");
	while (!feof(file) && count < nline)
        {
		text[0] = '\0';
		fgets(text,200,file);
		count++;
	}
	fclose(file);
	return count >= nline;
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

void write_out_landmarks(const string& outfeatures, const CLMTracker::CLM& clm_model)
{

	std::ofstream featuresFile;
	featuresFile.open(outfeatures.c_str());		

	if(featuresFile.is_open())
	{	
		// set landmart 68
		int n = clm_model.pdm.NumberOfPoints();

		if (n == 70)
		{
			featuresFile << "#version: 1" << endl;
			featuresFile << "#npoints: " << 68 << endl;
			featuresFile << "#{" << endl;

			for (int i = 0; i < 68; ++ i)
			{
				// Use matlab format, so + 1
				//featuresFile << clm_model.detected_landmarks.at<double>(i) + 2 << " " << clm_model.detected_landmarks.at<double>(i+n) + 2 << endl;
				featuresFile << clm_model.detected_landmarks.at<double>(i) << " " << clm_model.detected_landmarks.at<double>(i+n) << endl;
			}
		}
		else
		{
			featuresFile << "#version: 1" << endl;
			featuresFile << "#npoints: " << n << endl;
			featuresFile << "#{" << endl;

			for (int i = 0; i < n; ++ i)
			{
				// Use matlab format, so + 1
				//featuresFile << clm_model.detected_landmarks.at<double>(i) + 2 << " " << clm_model.detected_landmarks.at<double>(i+n) + 2 << endl;
				featuresFile << clm_model.detected_landmarks.at<double>(i) << " " << clm_model.detected_landmarks.at<double>(i+n) << endl;
			}
		}
		// temporary model_likelihood
		//featuresFile << clm_model.model_likelihood << endl;
		featuresFile << "#}" << endl;			
		featuresFile.close();
	}
}

void write_out_confidence(const string& confidence, const CLMTracker::CLM& clm_model, float normalized_error)
{
	std::ofstream confidenceFile;
	confidenceFile.open(confidence.c_str());		
	double cfd;
	cfd = (clm_model.model_likelihood + 1.5) * 0.5;
	if (cfd > 1.0) cfd = 1.0;
	else if (cfd < 0.0) cfd = 0.0;
	if(confidenceFile.is_open())
	{	
		//confidenceFile << normalized_error << endl;
		confidenceFile << cfd << endl;
		confidenceFile.close();
	}
}

void create_display_image(const Mat& orig, Mat& display_image, CLMTracker::CLM& clm_model)
{

	// preparing the visualisation image
	display_image = orig.clone();		

	// Creating a display image			
	Mat xs = clm_model.detected_landmarks(Rect(0, 0, 1, clm_model.detected_landmarks.rows/2));
	Mat ys = clm_model.detected_landmarks(Rect(0, clm_model.detected_landmarks.rows/2, 1, clm_model.detected_landmarks.rows/2));
	double min_x, max_x, min_y, max_y;

	cv::minMaxLoc(xs, &min_x, &max_x);
	cv::minMaxLoc(ys, &min_y, &max_y);

	double width = max_x - min_x;
	double height = max_y - min_y;
	//if (width < 5 || height < 5 || width*height < 100) {
	//	display_image = cv::Mat();
	//	return;
	//} 

	int minCropX = max((int)(min_x-width/3.0),0);
	int minCropY = max((int)(min_y-height/3.0),0);

	int widthCrop = min((int)(width*5.0/3.0), display_image.cols - minCropX - 1);
	int heightCrop = min((int)(height*5.0/3.0), display_image.rows - minCropY - 1);

	if(widthCrop <= 0 || heightCrop <=0) return;
	double scaling = 350.0/widthCrop;

	// first crop the image
	display_image = display_image(Rect((int)(minCropX), (int)(minCropY), (int)(widthCrop), (int)(heightCrop)));

	// now scale it
	cv::resize(display_image.clone(), display_image, Size(), scaling, scaling);

	// Make the adjustments to points
	xs = (xs - minCropX)*scaling;
	ys = (ys - minCropY)*scaling;

	Mat shape = clm_model.detected_landmarks.clone();

	xs.copyTo(shape(Rect(0, 0, 1, clm_model.detected_landmarks.rows/2)));
	ys.copyTo(shape(Rect(0, clm_model.detected_landmarks.rows/2, 1, clm_model.detected_landmarks.rows/2)));

	Draw(display_image, clm_model);

}

cv::Mat getCroppedIm(const Mat& orig, cv::Mat_<double> &oriLMs, cv::Mat_<double> &newLMs)
{	
	// Creating a display image			
	newLMs = oriLMs.clone();
	Mat xs = newLMs(Rect(0, 0, 1, oriLMs.rows/2));
	Mat ys = newLMs(Rect(0, oriLMs.rows/2, 1, oriLMs.rows/2));
	double min_x, max_x, min_y, max_y;

	cv::minMaxLoc(xs, &min_x, &max_x);
	cv::minMaxLoc(ys, &min_y, &max_y);
	printf("min max %f %f %f %f\n",min_x,max_x,min_y,max_y);
	double width = max_x - min_x;
	double height = max_y - min_y;

        if (width < 5 || height < 5 || width*height < 100) {
                return cv::Mat();
        }

	int minCropX = max((int)(min_x-width/3.0),0);
	int minCropY = max((int)(min_y-height/3.0),0);

	int widthCrop = min((int)(width*5.0f/3.0f), orig.cols - minCropX - 1);
	int heightCrop = min((int)(height*5.0f/3.0f), orig.rows - minCropY - 1);

	if(widthCrop <= 0 || heightCrop <=0) return cv::Mat();
	printf("widthCrop %d %d\n",widthCrop,widthCrop);
	double scaling = max(1.0f, std::sqrt(widthCrop*heightCrop/170000.0f));

	//int nrows = heightCrop/4 * 4;
	//int ncols = widthCrop/4 * 4;
	//if (nrows != read_image.rows || ncols != read_image.cols){
	//	read_image = read_image(Rect(0,0,ncols,nrows));
	//}
	// first crop the image
	cv::Mat display_image = orig(Rect((int)(minCropX), (int)(minCropY), (int)(widthCrop), (int)(heightCrop)));

	// now scale it
	if (scaling > 1)
		cv::resize(display_image.clone(), display_image, Size(), 1/scaling, 1/scaling);
	else
		display_image = display_image.clone();

	int nrows = display_image.rows/4 * 4;
	int ncols = display_image.cols/4 * 4;
	if (nrows != display_image.rows || ncols != display_image.cols){
		//cv::Mat tmp = display_image;
		display_image = display_image(Rect(0,0,ncols,nrows)).clone();
		//tmp.release();
	}
	xs = (xs - minCropX)/scaling;
	ys = (ys - minCropY)/scaling;
	return display_image;	
}


cv::Mat getCroppedIm(const Mat& orig, CLMTracker::CLM& clm_model, cv::Mat_<double> &newLMs)
{	
	// Creating a display image
	printf("getCroppedIm start\n");			
	newLMs = clm_model.detected_landmarks.clone();
	Mat xs = newLMs(Rect(0, 0, 1, clm_model.detected_landmarks.rows/2));
	Mat ys = newLMs(Rect(0, clm_model.detected_landmarks.rows/2, 1, clm_model.detected_landmarks.rows/2));
	double min_x, max_x, min_y, max_y;

	cv::minMaxLoc(xs, &min_x, &max_x);
	cv::minMaxLoc(ys, &min_y, &max_y);

	double width = max_x - min_x;
	double height = max_y - min_y;

	int minCropX = max((int)(min_x-width/3.0),0);
	int minCropY = max((int)(min_y-height/3.0),0);

	int widthCrop = min((int)(width*5.0f/3.0f), orig.cols - minCropX - 1);
	int heightCrop = min((int)(height*5.0f/3.0f), orig.rows - minCropY - 1);

        if (widthCrop < 5 || heightCrop < 5 || widthCrop*heightCrop < 100) {
                return cv::Mat();
        }

	double scaling = max(1.0f, std::sqrt(widthCrop*heightCrop/170000.0f));

	//int nrows = heightCrop/4 * 4;
	//int ncols = widthCrop/4 * 4;
	//if (nrows != read_image.rows || ncols != read_image.cols){
	//	read_image = read_image(Rect(0,0,ncols,nrows));
	//}
	// first crop the image
	cv::Mat display_image = orig(Rect((int)(minCropX), (int)(minCropY), (int)(widthCrop), (int)(heightCrop)));

	// now scale it
	if (scaling > 1)
		cv::resize(display_image.clone(), display_image, Size(), 1/scaling, 1/scaling);
	else
		display_image = display_image.clone();

	int nrows = display_image.rows/4 * 4;
	int ncols = display_image.cols/4 * 4;
	if (nrows != display_image.rows || ncols != display_image.cols){
		//cv::Mat tmp = display_image;
		display_image = display_image(Rect(0,0,ncols,nrows)).clone();
		//tmp.release();
	}
	xs = (xs - minCropX)/scaling;
	ys = (ys - minCropY)/scaling;
	printf("getCroppedIm done\n");	
	return display_image;	
}

int getStartF(string startFFile){
	FILE* file = fopen(startFFile.c_str(),"r");
	if (file == 0){
		return 0;
	}
	char str[200];
	char* pos[3];
	fgets(str,200,file);
	if (splittext(str,pos) == 0) {
		fclose(file);
		return 0;
	}
	int g = atoi(pos[0]);
	fclose(file);
	return g;
}

void saveStartF(string startFFile, int i){
	FILE* file = fopen(startFFile.c_str(),"w");
	if (file == 0){
		return;
	}
	fprintf(file, "%d\n",i);
	fclose(file);
}

void getRestrictList(vector<string> &list){
	FILE* file = fopen("bf.txt","r");
	if (file == 0) return;
	char tmp[200];
	while(!feof(file)){
		tmp[0] = '\0';
		fgets(tmp,200,file);
		if (strlen(tmp) < 2) continue;
		tmp[strlen(tmp)-1] = '\0';
		printf("%s-\n",tmp);
		list.push_back(/*string("C:/Users/Anh/CS0/landmarkP_sift/split1/test_1_A/landmark/") + */string(tmp));
	}
	fclose(file);
}

bool inList(string fname, vector<string> list){
	if (list.size() == 0) return true;
	for (int i=0;i<list.size();i++){
		if (strcmp(fname.c_str(),list[i].c_str()) == 0) return true;
	}
	return false;
}

int main (int argc, char **argv)
{
	//Convert arguments to more convenient vector form
	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line
	vector<string> files, depth_files, output_images, model_files, lm_files, pose_files, output_landmark_locations;
	vector<string> restrictList;
	string refDir;
	string badIn;
	getRestrictList(restrictList);
	float mstep = 0.001, threshSize = 80.0f, speed = 0.005f;
	bool preOpt = true;
	int numTri = 40;
	string baselFile, startFFile;

	// Bounding boxes for a face in each image (optional)
	vector<Rect_<double> > bounding_boxes;
	bool bJanus = false;
	vector<vector<float> > meta_infos;
	vector<int> image_types;
	string opath, olist;
	vector<string> inputLMFiles;
	bool continueWork = false;
	vector<int> similarFile;
	char text[200];
	int fromIndex, toIndex;

	CLMTracker::get_image_input_output_params(files, depth_files, output_landmark_locations, output_images,model_files, lm_files, pose_files, bounding_boxes, bJanus, meta_infos, image_types, arguments, opath, olist,inputLMFiles, similarFile,continueWork,refDir);	
	CLMTracker::get_image_alg_params(arguments, mstep, threshSize, speed, preOpt, numTri);
	printf("inputLMFiles %d\n", inputLMFiles.size());
	CLMTracker::get_basel_params(arguments, baselFile);
	CLMTracker::get_run_params(arguments, startFFile, badIn, fromIndex, toIndex);
	printf("input %s %s %d %d\n", startFFile.c_str(), baselFile.c_str(), fromIndex, toIndex);
	CLMTracker::CLMParameters clm_parameters(arguments);	
	// No need to validate detections, as we're not doing tracking
	clm_parameters.validate_detections = false;
	strcpy(text,baselFile.c_str());
	BaselFace::load_BaselFace_data(text);

	// The modules that are being used for tracking
	cout << "Loading the model" << endl;
	CLMTracker::CLM clm_model(clm_parameters.model_location,clm_parameters.face_detector_location);
	cout << "Model loaded" << endl;

	CascadeClassifier classifier(clm_parameters.face_detector_location);
	cout << "Classifier loaded!" << endl;
	bool visualise = false;
	int frontal_success = 0;
	int frontal_fail = 0;
	int profile_success = 0;
	int profile_fail = 0;	
	if (opath.length() > 0) {
		printf("Create outDir\n");
		boost::filesystem::path outdir(opath);
		boost::filesystem::create_directories(outdir);
	}
	int startF = 0;
	int numProc = 0;
	if (continueWork){
		printf("get startF\n");
		startF = getStartF(startFFile);
	}
	if (fromIndex > startF) startF = fromIndex;
	printf("startF %d\n",startF);
	std::ofstream listfile;
	listfile.open(olist.c_str());
	for (int i =0;i < files.size();i++){
		listfile << model_files[i] << std::endl;
	}
	listfile.close();
	FILE* bInFile = fopen(badIn.c_str(),"w");
	float fx, fy;
	fx = fy = 1000.0f;
	if (toIndex < 0 || toIndex > files.size())
		toIndex = files.size();
	// Do some image loading
	for(size_t i = startF; i < toIndex; i++)
	{
		//if (checkFile(model_files[i] + string(".alpha"),99) && checkFile(model_files[i] + string(".beta"),99) && checkFile(model_files[i] + string(".rend"),6)) {
		//	saveStartF(startFFile,i+1);	
		//	continue;
		//}
		if (!inList(model_files[i], restrictList)) continue;
		printf("%s\n", model_files[i].c_str());
		//if (similarFile[i] >= 0 && boost::filesystem::exists(model_files[similarFile[i]])){
		//	printf("similarFile %d\n",similarFile[i]);
		//	int ind = similarFile[i];
		//	if (output_landmark_locations.size()>0) {
		//		boost::filesystem::copy_file(output_landmark_locations[ind],output_landmark_locations[i],boost::filesystem::copy_option::overwrite_if_exists);

		//		string conf1 = output_landmark_locations.at(ind);
		//		conf1.replace(conf1.end() - 3, conf1.end(), "cfd");
		//		string conf2 = output_landmark_locations.at(i);
		//		conf2.replace(conf2.end() - 3, conf2.end(), "cfd");
		//		boost::filesystem::copy_file(conf1,conf2,boost::filesystem::copy_option::overwrite_if_exists);

		//	}
		//	if (output_images.size()>0)
		//		boost::filesystem::copy_file(output_images[ind],output_images[i],boost::filesystem::copy_option::overwrite_if_exists);
		//	if (model_files.size()>0)
		//		boost::filesystem::copy_file(model_files[ind],model_files[i],boost::filesystem::copy_option::overwrite_if_exists);
		//	if (lm_files.size()>0)
		//		boost::filesystem::copy_file(lm_files[ind],lm_files[i],boost::filesystem::copy_option::overwrite_if_exists);
		//	if (pose_files.size()>0)
		//		boost::filesystem::copy_file(pose_files[ind],pose_files[i],boost::filesystem::copy_option::overwrite_if_exists);
		//	if (output_images.size()>0)
		//		boost::filesystem::copy_file(output_images[ind],output_images[i],boost::filesystem::copy_option::overwrite_if_exists);
		//	continue;
		//}
		string file = files.at(i);
		printf("file %s-\n",file.c_str());

		// Loading image
		Mat read_image = imread(file, 1);
		printf("file stat %d-\n",read_image.data != NULL);
		printf("%d %d %d %d\n",read_image.rows,read_image.cols,read_image.channels(),read_image.type());
		if (read_image.data == NULL || read_image.rows < 1 || read_image.cols < 1 || read_image.channels() < 3 ) {
			fprintf(bInFile,"%s\n",file.c_str()); 
			continue;
		}
		float cx = read_image.cols / 2.0f;
		float cy = read_image.rows / 2.0f;
		// Loading depth file if exists (optional)
		Mat_<float> depth_image;

		if(depth_files.size() > 0)
		{
			string dFile = depth_files.at(i);
			Mat dTemp = imread(dFile, -1);
			dTemp.convertTo(depth_image, CV_32F);
		}

		// Making sure the image is in uchar grayscale
		Mat_<uchar> grayscale_image;
		printf("convert_to_grayscale\n");		
		convert_to_grayscale(read_image, grayscale_image);
		printf("done %d\n",inputLMFiles.size());

		if (inputLMFiles.size() > 0) {
			boost::filesystem::path pLM(inputLMFiles[i]);
			if (boost::filesystem::exists(pLM)) {
			printf("lm %s-\n",inputLMFiles[i].c_str());
			printf("lmo %s-\n",output_landmark_locations[i].c_str());
				//boost::filesystem::copy_file(inputLMFiles[i],output_landmark_locations[i],boost::filesystem::copy_option::overwrite_if_exists);
			printf("copied\n");

				string fcfd2 = output_landmark_locations.at(i);
				fcfd2.replace(fcfd2.end() - 3, fcfd2.end(), "cfd");
				string fcfd1 = inputLMFiles[i];
				fcfd1.replace(fcfd1.end() - 3, fcfd1.end(), "cfd");
				//boost::filesystem::copy_file(fcfd1,fcfd2,boost::filesystem::copy_option::overwrite_if_exists);
			printf("copied cfd\n");
				string inpose = inputLMFiles[i] + std::string(".pose");
				cv::Mat_<double> oriLMs;
				double cfd;
				printf("inputLMFiles[i] %s-\n",inputLMFiles[i].c_str());
				printf("fcfd1 %s-\n",fcfd1.c_str());
				Vec6d poseEstimateCLM0;
				if(CLMTracker::loadLM(inputLMFiles[i],fcfd1,inpose, oriLMs,cfd,poseEstimateCLM0) && !model_files.empty())
				{
					string model_file = model_files.at(i);
					string lm_file = lm_files.at(i);
					string pose_file = pose_files.at(i);
					
					printf("Init FaceService\n");
					FaceServices2 fservice;                      
					printf("Init FaceService done\n");

					float new_r[3], new_t[3];
					cv::Mat_<double> newLMs;
					cv::Mat cropped = getCroppedIm(read_image,oriLMs,newLMs);
					if (cropped.rows < 5 || cropped.cols < 5){
			                        fprintf(bInFile,"%s\n",file.c_str());
			                        continue;
					}
					sprintf(text,"%s_cropped.png",model_file.c_str());
					cv::imwrite(text,cropped);
					printf("setUp\n");
					fservice.setUp(cropped.cols,cropped.rows,1000.0f);
					//fservice.init(cropped,newLMs,1000.0f);
					//fservice.singleFrameReconSym(cropped,fservice.initR, fservice.initT,new_r, new_t,model_file,lm_file,pose_file);
					cv::Mat shape, tex;
					//fservice.config(mstep, threshSize, speed, preOpt, numTri);
					Vec6d poseEstimateCLM(0,0,0,0,0,0);
					cv::Mat_<int> lmVis = cv::Mat_<int>::ones(68,1);
					//if (cfd >= 0.65) {
					//for (int m=0;m<5;m++) {
					//	std::ostringstream ss;
    					//	ss << m;
                    			//	std::string model_file_tmp = model_file.substr(0, model_file.size() -4) + std::string("_") + ss.str() + std::string(".ply");
                    			//	std::string pose_file_tmp = lm_file.substr(0, pose_file.size() -5) + std::string("_") + ss.str() + std::string(".pose");
					fservice.singleFrameRecon(cropped, newLMs,poseEstimateCLM, cfd, lmVis, shape, tex, model_file,lm_file,pose_file,refDir);
					//}
					numProc++;
					//}
				}
			}
		}
		else {
			// if no pose defined we just use OpenCV
			if(bounding_boxes.empty())
			{
				printf("Detect face\n");
				// Detect faces in an image
				vector<Rect_<double> > face_detections;
				vector<float> meta_info;

				CLMTracker::DetectFaces(face_detections, grayscale_image, classifier);

				// Detect landmarks around detected faces
				
				// perform landmark detection for every face detected
				for(size_t face=0; face < face_detections.size(); ++face)
				{
					// if there are multiple detections go through them
					DetectLandmarksInImage(grayscale_image, depth_image, face_detections[face], clm_model, clm_parameters, bJanus, meta_info, 0);

					// Writing out the detected landmarks (in an OS independent manner)
					if(!output_landmark_locations.empty())
					{
						char name[100];
						// append detection number (in case multiple faces are detected)
						sprintf(name, "_det_%d", face);

						// Construct the output filename
						boost::filesystem::path slash("/");
						std::string preferredSlash = slash.make_preferred().string();

						boost::filesystem::path out_feat_path(output_landmark_locations.at(i));
						boost::filesystem::path dir = out_feat_path.parent_path();
						boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
						boost::filesystem::path ext = out_feat_path.extension();
						string outfeatures = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();
						write_out_landmarks(outfeatures, clm_model);
					}

					if(!model_files.empty())
					{
						string model_file = model_files.at(i);
						if(!model_file.empty())
						{
							char name[100];
							sprintf(name, "_det_%d", face);

							boost::filesystem::path slash("/");
							std::string preferredSlash = slash.make_preferred().string();

							// append detection number
							boost::filesystem::path out_feat_path(model_file);
							boost::filesystem::path dir = out_feat_path.parent_path();
							boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
							boost::filesystem::path ext = out_feat_path.extension();
							model_file = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();
							string lm_file = dir.string() + preferredSlash + fname.string() + string(name) + string("_lm") + ext.string();
							string pose_file = dir.string() + preferredSlash + fname.string() + string(name) + string(".pose");

							FaceServices2 fservice;
							float new_r[3], new_t[3];
							cv::Mat_<double> newLMs;
							cv::Mat cropped = getCroppedIm(read_image,clm_model,newLMs);

							sprintf(text,"%s_cropped.png",model_file.c_str());
							cv::imwrite(text,cropped);
							//fservice.init(cropped,newLMs,1000.0f);
							//fservice.singleFrameReconSym(cropped,fservice.initR, fservice.initT,new_r, new_t,model_file,lm_file,pose_file);
							cv::Mat shape, tex;
							fservice.setUp(cropped.cols,cropped.rows,1000.0f);
							//fservice.config(mstep, threshSize, speed, preOpt, numTri);
							int idx = clm_model.patch_experts.GetViewIdx(clm_model.params_global, 0);
							Vec6d poseEstimateCLM = GetPoseCLM(clm_model, fx, fy, cx, cy, clm_parameters);

							double cfd;
							cfd = (clm_model.model_likelihood + 1.5) * 0.5;
							if (cfd > 1.0) cfd = 1.0;
							else if (cfd < 0.0) cfd = 0.0;

							//if (cfd >= 0.65) {
							//for (int m=0;m<5;m++){
                                                //std::ostringstream ss;
                                                //ss << m;

                    //std::string model_file_tmp = model_file.substr(0, model_file.size() -4) + std::string("_") + ss.str() + std::string(".ply");
                    //std::string pose_file_tmp = lm_file.substr(0, pose_file.size() -5) + std::string("_") + ss.str() + std::string(".pose");
							fservice.singleFrameRecon(cropped, newLMs,poseEstimateCLM, cfd, clm_model.patch_experts.visibilities[0][idx], shape, tex, model_file,lm_file,pose_file,refDir);
							//}
							numProc++;
							//}
						}
					}
					// displaying detected landmarks
					Mat display_image;
					create_display_image(read_image, display_image, clm_model);

					if(visualise)
					{
						imshow("colour", display_image);
						cv::waitKey(1);
					}

					// Saving the display images (in an OS independent manner)
					if(!output_images.empty())
					{
						string outimage = output_images.at(i);
						if(!outimage.empty())
						{
							char name[100];
							sprintf(name, "_det_%d", face);

							boost::filesystem::path slash("/");
							std::string preferredSlash = slash.make_preferred().string();

							// append detection number
							boost::filesystem::path out_feat_path(outimage);
							boost::filesystem::path dir = out_feat_path.parent_path();
							boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
							boost::filesystem::path ext = out_feat_path.extension();
							outimage = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();

							imwrite(outimage, display_image);	
						}
					}

				}
			}
			else
			{
				if(bJanus == true)
				{
					int n = clm_model.pdm.NumberOfPoints();
					float normalized_error;
					bool bSuccess;
					if (image_types[i] == 1 || image_types[i] == 4)
					{
						// Have provided bounding boxes
						if (meta_infos[i][2] != 0 && meta_infos[i][4] < meta_infos[i][0]+meta_infos[i][2]/3)
						{
							DetectLandmarksInImage(grayscale_image, bounding_boxes[i], clm_model, clm_parameters, bJanus, meta_infos[i], 3);
						}
						else if (meta_infos[i][2] != 0 && meta_infos[i][6] > meta_infos[i][0]+meta_infos[i][2]*2/3)
						{
							DetectLandmarksInImage(grayscale_image, bounding_boxes[i], clm_model, clm_parameters, bJanus, meta_infos[i], 2);
						}
						else
						{
							DetectLandmarksInImage(grayscale_image, bounding_boxes[i], clm_model, clm_parameters, bJanus, meta_infos[i], 1);
						}
						float nose_x = clm_model.detected_landmarks.at<double>(33);
						float nose_y = clm_model.detected_landmarks.at<double>(33+n);
						float right_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(42) + clm_model.detected_landmarks.at<double>(45) );
						float right_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(42+n) + clm_model.detected_landmarks.at<double>(45+n) );
						float left_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(36) + clm_model.detected_landmarks.at<double>(39) );
						float left_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(36+n) + clm_model.detected_landmarks.at<double>(39+n) );

						float interocular_distance = sqrt( (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) * (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) 
							+ (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) * (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) );
						normalized_error = (sqrt( ( meta_infos[i][4] - right_eye_x) * ( meta_infos[i][4] - right_eye_x) 
							+ ( meta_infos[i][5] - right_eye_y) * ( meta_infos[i][5] - right_eye_y) ) +
							sqrt( ( meta_infos[i][8] - nose_x) * ( meta_infos[i][8] - nose_x) 
							+ ( meta_infos[i][9] - nose_y) * ( meta_infos[i][9] - nose_y) ) +
							sqrt( ( meta_infos[i][6] - left_eye_x) * ( meta_infos[i][6] - left_eye_x) 
							+ ( meta_infos[i][7] - left_eye_y) * ( meta_infos[i][7] - left_eye_y) )
							) / 3 / interocular_distance;

						float normalized_distance;
						normalized_distance = sqrt( ( meta_infos[i][6] - meta_infos[i][4]) * ( meta_infos[i][6] - meta_infos[i][4]) 
							+ (meta_infos[i][7] - meta_infos[i][5]) * (meta_infos[i][7] - meta_infos[i][5]) );
						if( sqrt( ( meta_infos[i][4] - right_eye_x) * ( meta_infos[i][4] - right_eye_x) 
							+ ( meta_infos[i][5] - right_eye_y) * ( meta_infos[i][5] - right_eye_y) ) < 0.2 * normalized_distance &&
							sqrt( ( meta_infos[i][8] - nose_x) * ( meta_infos[i][8] - nose_x) 
							+ ( meta_infos[i][9] - nose_y) * ( meta_infos[i][9] - nose_y) ) < 0.2 * normalized_distance &&					
							sqrt( ( meta_infos[i][6] - left_eye_x) * ( meta_infos[i][6] - left_eye_x) 
							+ ( meta_infos[i][7] - left_eye_y) * ( meta_infos[i][7] - left_eye_y) ) < 0.2 * normalized_distance ) 
						{
							printf("Detection Success - image number %d image type %d!\n", i, image_types[i]);
							frontal_success++;
							bSuccess = true;
						}
						else
						{
							printf("Warning: normalized error large - image number %d image type %d!\n", i, image_types[i]);
							frontal_fail++;
							bSuccess = false;
						}
					}
					else if (image_types[i] == 7 || image_types[i] == 8)
					{
						// Have provided bounding boxes
						if (meta_infos[i][2] != 0 && meta_infos[i][4] < meta_infos[i][0]+meta_infos[i][2]/3)
						{
							DetectLandmarksInImage(grayscale_image, bounding_boxes[i], clm_model, clm_parameters, bJanus, meta_infos[i], 3);
						}
						else if (meta_infos[i][2] != 0 && meta_infos[i][6] > meta_infos[i][0]+meta_infos[i][2]*2/3)
						{
							DetectLandmarksInImage(grayscale_image, bounding_boxes[i], clm_model, clm_parameters, bJanus, meta_infos[i], 2);
						}
						else
						{
							DetectLandmarksInImage(grayscale_image, bounding_boxes[i], clm_model, clm_parameters, bJanus, meta_infos[i], 1);
						}
						float nose_x = clm_model.detected_landmarks.at<double>(33);
						float nose_y = clm_model.detected_landmarks.at<double>(33+n);
						float right_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(42) + clm_model.detected_landmarks.at<double>(45) );
						float right_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(42+n) + clm_model.detected_landmarks.at<double>(45+n) );
						float left_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(36) + clm_model.detected_landmarks.at<double>(39) );
						float left_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(36+n) + clm_model.detected_landmarks.at<double>(39+n) );

						float interocular_distance = sqrt( (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) * (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) 
							+ (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) * (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) );
						normalized_error = (sqrt( ( meta_infos[i][4] - right_eye_x) * ( meta_infos[i][4] - right_eye_x) 
							+ ( meta_infos[i][5] - right_eye_y) * ( meta_infos[i][5] - right_eye_y) ) +
							sqrt( ( meta_infos[i][6] - left_eye_x) * ( meta_infos[i][6] - left_eye_x) 
							+ ( meta_infos[i][7] - left_eye_y) * ( meta_infos[i][7] - left_eye_y) )
							) / 2 / interocular_distance;

						float normalized_distance;
						normalized_distance = sqrt( ( meta_infos[i][6] - meta_infos[i][4]) * ( meta_infos[i][6] - meta_infos[i][4]) 
							+ (meta_infos[i][7] - meta_infos[i][5]) * (meta_infos[i][7] - meta_infos[i][5]) );
						if( sqrt( ( meta_infos[i][4] - right_eye_x) * ( meta_infos[i][4] - right_eye_x) 
							+ ( meta_infos[i][5] - right_eye_y) * ( meta_infos[i][5] - right_eye_y) ) < 0.2 * normalized_distance &&	
							sqrt( ( meta_infos[i][6] - left_eye_x) * ( meta_infos[i][6] - left_eye_x) 
							+ ( meta_infos[i][7] - left_eye_y) * ( meta_infos[i][7] - left_eye_y) ) < 0.2 * normalized_distance ) 
						{
							printf("Detection Success - image number %d image type %d!\n", i, image_types[i]);
							frontal_success++;
							bSuccess = true;
						}
						else
						{
							printf("Warning: normalized error large - image number %d image type %d!\n", i, image_types[i]);
							frontal_fail++;
							bSuccess = false;
						}
					}
					else if (image_types[i] == 2 || image_types[i] == 5)
					{
						// Have provided bounding boxes
						DetectLandmarksInImage(grayscale_image, bounding_boxes[i], clm_model, clm_parameters, bJanus, meta_infos[i], 2);
						float nose_x = clm_model.detected_landmarks.at<double>(33);
						float nose_y = clm_model.detected_landmarks.at<double>(33+n);
						float right_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(42) + clm_model.detected_landmarks.at<double>(45) );
						float right_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(42+n) + clm_model.detected_landmarks.at<double>(45+n) );
						float left_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(36) + clm_model.detected_landmarks.at<double>(39) );
						float left_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(36+n) + clm_model.detected_landmarks.at<double>(39+n) );

						float interocular_distance = sqrt( (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) * (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) 
							+ (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) * (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) );
						normalized_error = (sqrt( ( meta_infos[i][8] - nose_x) * ( meta_infos[i][8] - nose_x) 
							+ ( meta_infos[i][9] - nose_y) * ( meta_infos[i][9] - nose_y) ) +
							sqrt( ( meta_infos[i][6] - left_eye_x) * ( meta_infos[i][6] - left_eye_x) 
							+ ( meta_infos[i][7] - left_eye_y) * ( meta_infos[i][7] - left_eye_y) )
							) / 2 / interocular_distance;

						float normalized_distance;
						normalized_distance = sqrt( ( meta_infos[i][8] - meta_infos[i][6]) * ( meta_infos[i][8] - meta_infos[i][6]) 
							+ ( meta_infos[i][9] - meta_infos[i][7]) * ( meta_infos[i][9] - meta_infos[i][7]) );
						if( sqrt( ( meta_infos[i][8] - nose_x) * ( meta_infos[i][8] - nose_x) 
							+ ( meta_infos[i][9] - nose_y) * ( meta_infos[i][9] - nose_y) ) < 0.2 * normalized_distance &&					
							sqrt( ( meta_infos[i][6] - left_eye_x) * ( meta_infos[i][6] - left_eye_x) 
							+ ( meta_infos[i][7] - left_eye_y) * ( meta_infos[i][7] - left_eye_y) ) < 0.2 * normalized_distance ) 
						{
							printf("Detection Success - image number %d image type %d!\n", i, image_types[i]);
							profile_success++;
							bSuccess = true;
						}
						else
						{
							printf("Warning: normalized error large - image number %d image type %d!\n", i, image_types[i]);
							profile_fail++;
							bSuccess = false;
						}
					}
					else if (image_types[i] == 3 || image_types[i] == 6)
					{
						// Have provided bounding boxes
						DetectLandmarksInImage(grayscale_image, bounding_boxes[i], clm_model, clm_parameters, bJanus, meta_infos[i], 3);
						float nose_x = clm_model.detected_landmarks.at<double>(33);
						float nose_y = clm_model.detected_landmarks.at<double>(33+n);
						float right_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(42) + clm_model.detected_landmarks.at<double>(45) );
						float right_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(42+n) + clm_model.detected_landmarks.at<double>(45+n) );
						float left_eye_x = 0.5 * ( clm_model.detected_landmarks.at<double>(36) + clm_model.detected_landmarks.at<double>(39) );
						float left_eye_y = 0.5 * ( clm_model.detected_landmarks.at<double>(36+n) + clm_model.detected_landmarks.at<double>(39+n) );

						float interocular_distance = sqrt( (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) * (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(45)) 
							+ (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) * (clm_model.detected_landmarks.at<double>(36+n) - clm_model.detected_landmarks.at<double>(45+n)) );
						normalized_error = (sqrt( ( meta_infos[i][4] - right_eye_x) * ( meta_infos[i][4] - right_eye_x) 
							+ ( meta_infos[i][5] - right_eye_y) * ( meta_infos[i][5] - right_eye_y) ) +
							sqrt( ( meta_infos[i][8] - nose_x) * ( meta_infos[i][8] - nose_x) 
							+ ( meta_infos[i][9] - nose_y) * ( meta_infos[i][9] - nose_y) )
							) / 2 / interocular_distance;

						float normalized_distance;
						normalized_distance = sqrt( ( meta_infos[i][4] - meta_infos[i][8]) * ( meta_infos[i][4] - meta_infos[i][8]) 
							+ ( meta_infos[i][5] - meta_infos[i][9]) * ( meta_infos[i][5] - meta_infos[i][9]) );
						if( sqrt( ( meta_infos[i][4] - right_eye_x) * ( meta_infos[i][4] - right_eye_x) 
							+ ( meta_infos[i][5] - right_eye_y) * ( meta_infos[i][5] - right_eye_y) ) < 0.2 * normalized_distance &&
							sqrt( ( meta_infos[i][8] - nose_x) * ( meta_infos[i][8] - nose_x) 
							+ ( meta_infos[i][9] - nose_y) * ( meta_infos[i][9] - nose_y) ) < 0.2 * normalized_distance ) 
						{
							printf("Detection Success - image number %d image type %d!\n", i, image_types[i]);
							profile_success++;
							bSuccess = true;
						}
						else
						{
							printf("Warning: normalized error large - image number %d image type %d!\n", i, image_types[i]);
							profile_fail++;
							bSuccess = false;
						}
					}

					// Writing out the detected landmarks
					if(!output_landmark_locations.empty())
					{
						string outfeatures = output_landmark_locations.at(i);
						write_out_landmarks(outfeatures, clm_model);

						string conf = output_landmark_locations.at(i);
						conf.replace(conf.end() - 3, conf.end(), "cfd");
						write_out_confidence(conf, clm_model, normalized_error);
					}

					if(!model_files.empty())
					{
						string model_file = model_files.at(i);
						string lm_file = lm_files.at(i);
						string pose_file = pose_files.at(i);
						printf("model %s\n",model_file.c_str());

						FaceServices2 fservice;
						float new_r[3], new_t[3];
						cv::Mat_<double> newLMs;
						cv::Mat cropped = getCroppedIm(read_image,clm_model,newLMs);		
						sprintf(text,"%s_cropped.png",model_file.c_str());
						cv::imwrite(text,cropped);
						//fservice.init(cropped,newLMs,1000.0f);
						//fservice.singleFrameReconSym(cropped,fservice.initR, fservice.initT,new_r, new_t,model_file,lm_file, pose_file);

						cv::Mat shape, tex;
						fservice.setUp(cropped.cols,cropped.rows,1000.0f);
						//fservice.config(mstep, threshSize, speed, preOpt, numTri);
						int idx = clm_model.patch_experts.GetViewIdx(clm_model.params_global, 0);
						Vec6d poseEstimateCLM = GetPoseCLM(clm_model, fx, fy, cx, cy, clm_parameters);

						double cfd;
						cfd = (clm_model.model_likelihood + 1.5) * 0.5;
						if (cfd > 1.0) cfd = 1.0;
						else if (cfd < 0.0) cfd = 0.0;

						//if (cfd >= 0.65) {
						//for (int m=0;m<5;m++){                                                
						//std::ostringstream ss;
                                                //ss << m;

                    //std::string model_file_tmp = model_file.substr(0, model_file.size() -4) + std::string("_") + ss.str() + std::string(".ply");
                    //std::string pose_file_tmp = lm_file.substr(0, pose_file.size() -5) + std::string("_") + ss.str() + std::string(".pose");
						fservice.singleFrameRecon(cropped, newLMs,poseEstimateCLM, cfd, clm_model.patch_experts.visibilities[0][idx], shape, tex, model_file,lm_file,pose_file,refDir);
						//}
						numProc++;
						//}
					}
				}
				else
				{
					vector<float> meta_info;

					// Have provided bounding boxes
					DetectLandmarksInImage(grayscale_image, bounding_boxes[i], clm_model, clm_parameters, bJanus, meta_info, 0);

					// Writing out the detected landmarks
					if(!output_landmark_locations.empty())
					{
						string outfeatures = output_landmark_locations.at(i);
						write_out_landmarks(outfeatures, clm_model);
					}

					if(!model_files.empty())
					{
						string model_file = model_files.at(i);
						string lm_file = lm_files.at(i);
						string pose_file = pose_files.at(i);
						printf("model %s\n",model_file.c_str());

						FaceServices2 fservice;
						float new_r[3], new_t[3];
						cv::Mat_<double> newLMs;
	printf("getCroppedIm %d\n");
						cv::Mat cropped = getCroppedIm(read_image,clm_model,newLMs);
						//cv::imwrite("new.png",display_image);
						//cv::Mat tmp = display_image.clone();
						//for (int i=0;i<clm_model.detected_landmarks.rows/2;i++){
						//	circle(tmp,Point(newLMs.at<double>(0,i),newLMs.at<double>(0,i+clm_model.detected_landmarks.rows/2)),2, cv::Scalar(255,0,0));
						//}
						//imwrite("newl.png",tmp);				
						sprintf(text,"%s_cropped.png",model_file.c_str());
						cv::imwrite(text,cropped);
						//fservice.init(cropped,newLMs,1000.0f);
						//fservice.singleFrameReconSym(cropped,fservice.initR, fservice.initT,new_r, new_t,model_file,lm_file,pose_file);
						cv::Mat shape, tex;
						fservice.setUp(cropped.cols,cropped.rows,1000.0f);
						//fservice.config(mstep, threshSize, speed, preOpt, numTri);
						int idx = clm_model.patch_experts.GetViewIdx(clm_model.params_global, 0);
						Vec6d poseEstimateCLM = GetPoseCLM(clm_model, fx, fy, cx, cy, clm_parameters);


						double cfd;
						cfd = (clm_model.model_likelihood + 1.5) * 0.5;
						if (cfd > 1.0) cfd = 1.0;
						else if (cfd < 0.0) cfd = 0.0;
						//if (cfd >= 0.65) {
						//for (int m=0;m<5;m++){
                                                //std::ostringstream ss;
                                                //ss << m;

                    				//std::string model_file_tmp = model_file.substr(0, model_file.size() -4) + std::string("_") + ss.str() + std::string(".ply");
                    				//std::string pose_file_tmp = lm_file.substr(0, pose_file.size() -5) + std::string("_") + ss.str() + std::string(".pose");
						fservice.singleFrameRecon(cropped, newLMs,poseEstimateCLM, cfd, clm_model.patch_experts.visibilities[0][idx], shape, tex, model_file,lm_file,pose_file,refDir);
						//}
						numProc++;
						//}
					}
				}

				// displaying detected stuff
				Mat display_image;
				create_display_image(read_image, display_image, clm_model);

				if(visualise)
				{
					imshow("colour", display_image);
					cv::waitKey(1);
				}

				if(!output_images.empty())
				{
					string outimage = output_images.at(i);
					if(!outimage.empty())
					{
						imwrite(outimage, display_image);	
					}
				}
			}
		}
		saveStartF(startFFile,i+1);	
		if (numProc >= 36) return 0;
	}

	saveStartF(startFFile,-1);
/*
	std::ofstream resultFile;
	string outresult = opath + ".txt";
	resultFile.open(outresult.c_str());		
	int frontal_total = frontal_success + frontal_fail;
	int profile_total = profile_success + profile_fail;

	if(resultFile.is_open())
	{	
		resultFile << "frontal_success: " << frontal_success << endl;
		resultFile << "frontal_fail: " << frontal_fail << endl;
		resultFile << "frontal_total: " << frontal_total << endl;
		resultFile << "profile_success: " << profile_success << endl;
		resultFile << "profile_fail: " << profile_fail << endl;	
		resultFile << "profile_total: " << profile_total << endl;		
		resultFile.close();
	}
*/
	fclose(bInFile);
	return 0;
}

