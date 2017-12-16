/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "FaceServices2.h"
#include <fstream>
#include "opencv2/contrib/contrib.hpp"
#include <Eigen/SparseLU>
//#include <Eigen/SPQRSupport>
//#include <omp.h>

using namespace std;
using namespace cv;

cv::Mat* FaceServices2::symSPC = 0;
cv::Mat* FaceServices2::symTPC = 0;

FaceServices2::FaceServices2(void)
{
	prevEF = 1000;
	mstep = 0.0001;
	countFail = 0;
	maxVal = 4;
	mlambda = 0.005;

	for (int i=0;i<TEXEDGE_ORIENTATION_REG_NUM;i++) 
		TexAngleCenter[i] = 2*i*M_PI/TEXEDGE_ORIENTATION_REG_NUM - M_PI;
	texEdgeDist = conEdgeDist = texEdgeDistDX = texEdgeDistDY = conEdgeDistDX = conEdgeDistDY = 0;

	texEdge = conEdge = 0;
	festimator.getTextureEdgeIndices(texEdgeIndices);

	if (FaceServices2::symSPC == 0){
		FaceServices2::symSPC = new cv::Mat(3,99,CV_32F);
		FaceServices2::symTPC = new cv::Mat(3,99,CV_32F);
		for (int j=0;j<3;j++){
		    for (int i=0;i<99;i++) {
		        FaceServices2::symSPC->at<float>(j,i) = BaselFace::BaselFace_symSPC[99*j+i];
		        FaceServices2::symTPC->at<float>(j,i) = BaselFace::BaselFace_symTPC[99*j+i];
		    }
		}

	}
}

void FaceServices2::setUp(int w, int h, float f){
	memset(_k,0,9*sizeof(float));
	_k[8] = 1;
	_k[0] = -f;
	_k[4] = f;
	_k[2] = w/2.0f;
	_k[5] = h/2.0f;
}

bool FaceServices2::projectCheckVis(FImRenderer* im_render, cv::Mat shape, float* r, float *t, cv::Mat refDepth, bool* &visible){
	float zNear_ = im_render->zNear;
	float zFar_ = im_render->zFar;

	cv::Mat k_m( 3, 3, CV_32F, _k );

	int nV = shape.rows;
	if (visible == 0) visible = new bool[shape.rows];
	cv::Mat rVec( 3, 1, CV_32F, r );
	cv::Mat tVec( 3, 1, CV_32F, t );
	cv::Mat rMat;
	cv::Rodrigues(rVec, rMat);
	cv::Mat new3D = rMat * shape.t() + cv::repeat(tVec,1,nV);

	for (int i=0;i<nV;i++){
		visible[i] = false;
		float Z = new3D.at<float>(2,i);
		float x = -new3D.at<float>(0,i)/Z*_k[4] + _k[2];
		float y = new3D.at<float>(1,i)/Z*_k[4] + _k[5];
		if (x > 0 && y > 0 & x < refDepth.cols-1 && y <refDepth.rows-1) {
			for (int dx =-1;dx<2;dx++){
				for (int dy =-1;dy<2;dy++){
					float dd = refDepth.at<float>(y+dy,x+dx);
					dd = - zNear_*zFar_   / ( zFar_ - dd * ( zFar_ - zNear_ ));
					if (fabs(Z - dd) < 5){
						visible[i] = true;
					}
				}
			}
		}
	}

	return true;
}

std::vector<cv::Point2f> FaceServices2::projectCheckVis2(FImRenderer* im_render, cv::Mat shape, float* r, float *t, cv::Mat refDepth, bool* &visible){
	float zNear_ = im_render->zNear;
	float zFar_ = im_render->zFar;

	cv::Mat k_m( 3, 3, CV_32F, _k );
	std::vector<cv::Point2f> out;
	int nV = shape.rows;
	if (visible == 0) visible = new bool[shape.rows];
	cv::Mat rVec( 3, 1, CV_32F, r );
	cv::Mat tVec( 3, 1, CV_32F, t );
	cv::Mat rMat;
	cv::Rodrigues(rVec, rMat);
	cv::Mat new3D = rMat * shape.t() + cv::repeat(tVec,1,nV);

	for (int i=0;i<nV;i++){
		visible[i] = false;
		float Z = new3D.at<float>(2,i);
		float x = -new3D.at<float>(0,i)/Z*_k[4] + _k[2];
		float y = new3D.at<float>(1,i)/Z*_k[4] + _k[5];
		out.push_back(cv::Point2f(x,y));
		if (x > 0 && y > 0 & x < refDepth.cols-1 && y <refDepth.rows-1) {
			for (int dx =-1;dx<2;dx++){
				for (int dy =-1;dy<2;dy++){
					float dd = refDepth.at<float>(y+dy,x+dx);
					dd = - zNear_*zFar_   / ( zFar_ - dd * ( zFar_ - zNear_ ));
					if (fabs(Z - dd) < 5){
						visible[i] = true;
					}
				}
			}
		}
	}

	return out;
}

bool FaceServices2::singleFrameRecon(cv::Mat colorIm, cv::Mat lms,cv::Vec6d poseCLM, float conf,cv::Mat lmVis, cv::Mat &shape, cv::Mat &tex, string model_file, string lm_file, string pose_file, string refDir){
	printf("set num of thread\n");
  	omp_set_num_threads(1);
	float renderParams[RENDER_PARAMS_COUNT];
	float renderParams2[RENDER_PARAMS_COUNT];
	Mat k_m(3,3,CV_32F,_k);
	//BaselFaceEstimator festimator;
	BFMParams params;
	params.init();
	printf("prepareEdgeDistanceMaps\n");
	prepareEdgeDistanceMaps(colorIm);
	printf("prepareEdge DistanceMaps done\n");
	Mat alpha = cv::Mat::zeros(20,1,CV_32F);
	Mat beta = cv::Mat::zeros(20,1,CV_32F);
	Mat exprW = cv::Mat::zeros(29,1,CV_32F);
	Mat alpha_bk, beta_bk, exprW_bk;
	shape = festimator.getShape(alpha);
	tex = festimator.getTexture(beta);
	printf("3D landmarks\n");
	Mat landModel0 = festimator.getLM(shape,poseCLM(4));
	float bCost, cCost, fCost;
	int bestIter = 0;
	bCost = 10000.0f;
	//write_plyFloat("visLM0.ply",landModel0.t());
	std::vector<int> lmVisInd;
	for (int i=0;i<60;i++){
		if (lmVis.at<int>(i)){
			if (/*(i< 17 || i> 26) &&*/ (i > 16 || abs(poseCLM(4)) <= M_PI/10 || (poseCLM(4) > M_PI/10 && i > 7) || (poseCLM(4) < -M_PI/10 && i < 9)))
				lmVisInd.push_back(i);
		}
	}
	cv::Mat tmpIm = colorIm.clone();
	
	printf("2D landmarks\n");
	if (lmVisInd.size() < 8) return false;
	Mat landModel = cv::Mat( lmVisInd.size(),3,CV_32F);
	Mat landIm = cv::Mat( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landModel.at<float>(i,0) = landModel0.at<float>(ind,0);
		landModel.at<float>(i,1) = landModel0.at<float>(ind,1);
		landModel.at<float>(i,2) = landModel0.at<float>(ind,2);
		landIm.at<float>(i,0) = lms.at<double>(ind);
		landIm.at<float>(i,1) = lms.at<double>(ind+landModel0.rows);
		//cv::circle(tmpIm,Point(landIm.at<float>(i,0),landIm.at<float>(i,1)),1,Scalar(255,0,0),1);
	}
	//imwrite("visLM.png",tmpIm);
	//write_plyFloat("visLM.ply",landModel.t());
	//getchar();

	printf("estimatedPose\n");
	cv::Mat vecR, vecT;
	festimator.estimatePose3D(landModel,landIm,k_m,vecR,vecT);
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_R+i] = vecR.at<float>(i,0);
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_T+i] = vecT.at<float>(i,0);
	printf("pose: %f %f %f %f %f %f\n",vecR.at<float>(0,0),vecR.at<float>(1,0),vecR.at<float>(2,0),vecT.at<float>(0,0),vecT.at<float>(1,0),vecT.at<float>(2,0));
	if (-vecT.at<float>(2,0) > 9900) return false;
	memcpy(renderParams,params.initR,sizeof(float)*RENDER_PARAMS_COUNT);

	cv::Mat faces = festimator.getFaces() - 1;
	cv::Mat faces_fill = festimator.getFaces_fill() - 1;
	cv::Mat colors;

	im_render = new FImRenderer(cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3));
	im_render->loadMesh(shape,tex,faces_fill);
	memset(params.sF,0,sizeof(float)*NUM_EXTRA_FEATURES);

	params.sI = 0.0;
	params.sF[FEATURES_LANDMARK] = 8.0f;
	//params.sF[FEATURES_TEXTURE_EDGE] = 4.0f;
	char text[200];
	Mat alpha0, beta0;
	int iter=0;
	int badCount = 0;
	double time;
	int M = 99;
	//params.sF[FEATURES_TEXTURE_EDGE] = 6;
	//params.sF[FEATURES_CONTOUR_EDGE] = 6;
	//params.optimizeAB[0] = params.optimizeAB[1] = false;
	memset(params.doOptimize,true,sizeof(bool)*6);
	if (refDir.length() == 0){
		for (;iter<10000;iter++) {
			if (iter%50 == 0) {
				//params.sI += 0.5f;
				//if ( params.sF >= 1.0f)
				//params.sF -= 1.0f;
				//sprintf(text,"tmp_%05d.png",iter);
				//renderFace(text, colorIm,landIm,false,  alpha, beta, faces, renderParams,exprW );
				updateTriangles(colorIm,faces,false,  alpha, renderParams, params, exprW );
				cCost = updateHessianMatrix(false, alpha,beta,renderParams,faces,colorIm,lmVisInd,landIm,params, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
				//if (bCost > cEF){
				//	alpha_bk.release();
				//	alpha_bk = alpha.clone();
				//	bCost = cEF;
				//	badCount = 0;
				//	memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
				//}
				//else {
				//	badCount++;
				//}
				//if (badCount > 1) break;
				//std::cout << "hess " << params.hessDiag << std::endl;
				prevEF = cEF;
				//getchar();
			}
			sno_step2(false, alpha, beta, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW);
			//getchar();
			//params.sF -= 0.002;
		}
		//alpha = alpha_bk.clone();
		//memcpy(renderParams,renderParams2,sizeof(float)*RENDER_PARAMS_COUNT);

		iter = 10000;
		params.optimizeAB[0] = true;
		params.sF[FEATURES_LANDMARK] = 2;
		params.sF[FEATURES_TEXTURE_EDGE] = 0; //10
		params.sF[FEATURES_CONTOUR_EDGE] = 15;
		badCount = 0;
	//mstep = 0.01;
		for (;iter<15000;iter++) {
			if (iter%20 == 0) {
				//params.sI += 0.5f;
				//if ( params.sF >= 1.0f)
				//params.sF -= 1.0f;
				//sprintf(text,"tmp_%05d.png",iter);
				//renderFace(text, colorIm,landIm,false,  alpha, beta, faces, renderParams,exprW );
				updateTriangles(colorIm,faces,false,  alpha, renderParams, params,exprW );
				cCost = updateHessianMatrix(false, alpha,beta,renderParams,faces,colorIm,lmVisInd,landIm,params,exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
				
				//if (bCost > cEF){
				//	alpha_bk.release();
				//	alpha_bk = alpha.clone();
				//	bCost = cEF;
				//	badCount = 0;
				//	memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
				//}
				//else {
				//	badCount++;
				//}
				//if (badCount > 1) break;
				//std::cout << "hess " << params.hessDiag << std::endl;
				prevEF = cEF;
				//getchar();
			}
			sno_step2(false, alpha, beta, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW);
			//getchar();
			//params.sF -= 0.002;
		}
		//alpha = alpha_bk.clone();
		//memcpy(renderParams,renderParams2,sizeof(float)*RENDER_PARAMS_COUNT);

		
	mstep = 0.001;
		bCost = 10000;
		iter = 15000;
		//beta = beta_bk.clone();
		//shape = festimator.getShape(alpha);
		//tex = festimator.getTexture(beta);
		//sprintf(text,"%s_10.ply",model_file.c_str());
		//write_plyFloat(text,shape,tex,faces);
		////getchar();
		//sprintf(text,"%s_10.alpha",model_file.c_str());
		//FILE* ff=fopen(text,"w");
		//for (int i=0;i<alpha.rows;i++) fprintf(ff,"%f\n",alpha.at<float>(i,0));
		//fclose(ff);
		//sprintf(text,"%s_10.beta",model_file.c_str());
		//ff=fopen(text,"w");
		//for (int i=0;i<beta.rows;i++) fprintf(ff,"%f\n",beta.at<float>(i,0));
		//fclose(ff);
		mstep = 0.001;
		params.optimizeAB[0] = false;
		params.optimizeAB[1] = false;
		params.optimizeExpr = false;
		params.sF[FEATURES_LANDMARK] = params.sF[FEATURES_TEXTURE_EDGE] = params.sF[FEATURES_CONTOUR_EDGE] = 0;
		params.sI = 15.0;
		badCount = 0;
		params.computeEI = true;
		memset(params.doOptimize,false,sizeof(bool)*6);
		memset(params.doOptimize+6,true,sizeof(bool)*(RENDER_PARAMS_COUNT-6));
		//params.doOptimize[RENDER_PARAMS_AMBIENT+1] = params.doOptimize[RENDER_PARAMS_AMBIENT+2] = false;
		//params.doOptimize[RENDER_PARAMS_DIFFUSE+1] = params.doOptimize[RENDER_PARAMS_DIFFUSE+2] = false;
	printf("loop\n");
		for (;iter<17000;iter++) {
			if (iter%50 == 0) {
				//params.sI += 0.5f;
				//if ( params.sF >= 1.0f)
				//params.sF -= 1.0f;
				//sprintf(text,"tmp_%05d.png",iter);
				//renderFace(text, colorIm,landIm,false,  alpha, beta, faces, renderParams,exprW );
				updateTriangles(colorIm,faces,false,  alpha, renderParams, params,exprW );
				cCost = updateHessianMatrix(false, alpha,beta,renderParams,faces,colorIm,lmVisInd,landIm,params,exprW);
				if (countFail > 40) {
					countFail = 0;
					break;
				}
				
				//if (bCost > cEF){
				//	alpha_bk.release();
				//	alpha_bk = alpha.clone();
				//	bCost = cEF;
				//	badCount = 0;
				//	memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
				//}
				//else {
				//	badCount++;
				//}
				//if (badCount > 1) break;
				//std::cout << "hess " << params.hessDiag << std::endl;
				prevEF = cEF;
				//getchar();
			}
			sno_step2(false, alpha, beta, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW);
			//getchar();
			//params.sF -= 0.002;
		}

		iter = 17000;
		alpha0 = alpha.clone();
		beta0 = beta.clone();
		alpha = cv::Mat::zeros(M,1,CV_32F);
		beta = cv::Mat::zeros(M,1,CV_32F);
		for (int i=0;i<alpha0.rows; i++) alpha.at<float>(i,0) = alpha0.at<float>(i,0);
		for (int i=0;i<beta0.rows; i++) beta.at<float>(i,0) = beta0.at<float>(i,0);
		alpha_bk = alpha.clone();
		beta_bk = beta.clone();
		exprW_bk = exprW.clone();
		badCount = 0;

		//for (;iter<3000;iter++) {
		//	if (iter%500 == 0) {
		//		sprintf(text,"tmp_%05d.png",iter);
		//		renderFace(text, colorIm, alpha, beta, faces, renderParams );
		//		updateTriangles(colorIm,faces, alpha, renderParams, params );
		//		updateHessianMatrix(alpha,beta,renderParams,faces,colorIm,lmVisInd,landIm,params);
		//		std::cout << "hess " << params.hessDiag << std::endl;
		//		getchar();
		//	}
		//	sno_step(alpha, beta, renderParams, faces,colorIm,lmVisInd,landIm,params);
		//	//getchar();
		//	//params.sF -= 0.002;
		//	//params.sI += 0.002;
		//}
		params.sI = 15.0f;
		params.sF[FEATURES_TEXTURE_EDGE] = 0;  //3.5
		params.sF[FEATURES_CONTOUR_EDGE] = 5.0f;
		//params.sF[FEATURES_LANDMARK] = 5.0f;
		params.optimizeAB[0] = params.optimizeAB[1] = true;
		params.optimizeExpr = true;
		memset(params.doOptimize,true,sizeof(bool)*RENDER_PARAMS_COUNT);
		//params.doOptimize[RENDER_PARAMS_AMBIENT+1] = params.doOptimize[RENDER_PARAMS_AMBIENT+2] = false;
		//params.doOptimize[RENDER_PARAMS_DIFFUSE+1] = params.doOptimize[RENDER_PARAMS_DIFFUSE+2] = false;
		//params.doOptimize[RENDER_PARAMS_GAIN] = params.doOptimize[RENDER_PARAMS_GAIN+1] = params.doOptimize[RENDER_PARAMS_GAIN+2] = false;
		time = (double)cv::getTickCount();
		for (;iter<20000;iter++) {
			//if (iter % 4000 == 0){
			//	M += 10;
			//	alpha0 = alpha.clone();
			//	beta0 = beta.clone();
			//	alpha = cv::Mat::zeros(M,1,CV_32F);
			//	beta = cv::Mat::zeros(M,1,CV_32F);
			//	for (int i=0;i<alpha0.rows; i++) alpha.at<float>(i,0) = alpha0.at<float>(i,0);
			//	for (int i=0;i<beta0.rows; i++) beta.at<float>(i,0) = beta0.at<float>(i,0);
			//}

			if (iter%250 == 0) {
				//if (iter>4000) memset(params.doOptimize,false,sizeof(bool)*RENDER_PARAMS_COUNT);
				//if (iter>=18500 && conf < 0.95) {
				//	params.sF[FEATURES_LANDMARK] = (conf - 0.85)/0.1 * 5.0f;
				//	//printf("conf %f %f\n",conf,params.sF);
				//	if (params.sF[FEATURES_LANDMARK] < 0) 
				//		params.sF[FEATURES_LANDMARK] = 0;
				//}
				//if (iter>=15000) params.sF = 0;
				//if (iter>=3000) params.sF = 0;
				//if ( params.sF >= 1.0f)
				//params.sI += 1.0f;
				//if ( params.sF >= 5.0f)
				//params.sF -= 0.5f;
				//sprintf(text,"tmp_%05d.png",iter);
				//renderFace(text, colorIm,landIm,false,  alpha, beta, faces, renderParams, exprW );
				updateTriangles(colorIm,faces,false,  alpha, renderParams, params, exprW );

				//write_FaceArea("area.ply", shape, faces, params.triVis, params.triAreas);
				//write_FaceShadow("shadow.ply", shape, faces, params.triVis, params.triNoShadow);
				//getchar();
				//double time = (double)cv::getTickCount();
				cCost = updateHessianMatrix(false, alpha,beta,renderParams,faces,colorIm,lmVisInd,landIm,params, exprW,false);
				if (iter == 17000) fCost = cCost;
				if (bCost > cCost || bestIter <= 17000){
					alpha_bk.release(); beta_bk.release(); exprW_bk.release();
					alpha_bk = alpha.clone();
					beta_bk = beta.clone();
					exprW_bk = exprW.clone();
					bCost = cCost;
					badCount = 0;
					bestIter = iter;
					memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
				}
				else {
					badCount++;
				}
				//getchar();
				//if (badCount > 2) break;
				//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
				//std::cout << "Times passed: " << time << std::endl;
				//time = (double)cv::getTickCount();
				//std::cout << "hess " << params.hessDiag << std::endl;
				//getchar();
			}
			sno_step2(false, alpha, beta, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW);
			//getchar();
			//params.sF -= 0.002;
		}

		//sprintf(text,"tmp_%05d.png",iter);
		//renderFace(text, colorIm,landIm,false,  alpha, beta, faces, renderParams );
		//alpha = alpha_bk.clone();
		//beta = beta_bk.clone();
		//memcpy(renderParams,renderParams2,sizeof(float)*RENDER_PARAMS_COUNT);
		updateTriangles(colorIm,faces,false,  alpha, renderParams, params,exprW );

		//double time = (double)cv::getTickCount();
		cCost = updateHessianMatrix(false, alpha,beta,renderParams,faces,colorIm,lmVisInd,landIm,params,exprW);
		if (bCost > cCost){
			alpha_bk.release(); beta_bk.release(); exprW_bk.release();
			alpha_bk = alpha.clone();
			beta_bk = beta.clone();
			exprW_bk = exprW.clone();
			bCost = cCost;
			badCount = 0;
			bestIter = iter;
			memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
		}
	}
	else {
		int EM = 29;
		loadReference(refDir, model_file, alpha, beta, renderParams, M, exprW, EM);

		//std::cout << "alpha " << alpha << std::endl;
		//std::cout << "beta " << beta << std::endl;
		//getchar();
		params.sI = 15.0f;
		params.sF[FEATURES_LANDMARK] = 8.0f;
		if (conf < 0.95) {
			params.sF[FEATURES_LANDMARK] = (conf - 0.85)/0.1 * 8.0f;
			//printf("conf %f %f\n",conf,params.sF);
			if (params.sF[FEATURES_LANDMARK] < 0) 
				params.sF[FEATURES_LANDMARK] = 0;
		}
		params.computeEI = true;
		params.optimizeAB[0] = params.optimizeAB[1] = true;
		memset(params.doOptimize,true,sizeof(bool)*RENDER_PARAMS_COUNT);
		//params.doOptimize[RENDER_PARAMS_AMBIENT+1] = params.doOptimize[RENDER_PARAMS_AMBIENT+2] = false;
		//params.doOptimize[RENDER_PARAMS_DIFFUSE+1] = params.doOptimize[RENDER_PARAMS_DIFFUSE+2] = false;
		//params.doOptimize[RENDER_PARAMS_GAIN] = params.doOptimize[RENDER_PARAMS_GAIN+1] = params.doOptimize[RENDER_PARAMS_GAIN+2] = false;
		//time = (double)cv::getTickCount();
		for (iter = 0;iter<1000;iter++) {
			//if (iter % 4000 == 0){
			//	M += 10;
			//	alpha0 = alpha.clone();
			//	beta0 = beta.clone();
			//	alpha = cv::Mat::zeros(M,1,CV_32F);
			//	beta = cv::Mat::zeros(M,1,CV_32F);
			//	for (int i=0;i<alpha0.rows; i++) alpha.at<float>(i,0) = alpha0.at<float>(i,0);
			//	for (int i=0;i<beta0.rows; i++) beta.at<float>(i,0) = beta0.at<float>(i,0);
			//}

			if (iter%250 == 0) {
				//std::cout << "alpha " << alpha << std::endl;
				//std::cout << "beta " << beta << std::endl;
				//if (iter>4000) memset(params.doOptimize,false,sizeof(bool)*RENDER_PARAMS_COUNT);

				//if (iter>=15000) params.sF = 0;
				//if (iter>=3000) params.sF = 0;
				//if ( params.sF >= 1.0f)
				//params.sI += 1.0f;
				//if ( params.sF >= 5.0f)
				//params.sF -= 0.5f;
				//sprintf(text,"tmp_%05d.png",iter);
				//renderFace(text, colorIm,landIm,false,  alpha, beta, faces, renderParams );
				updateTriangles(colorIm,faces,false,  alpha, renderParams, params,exprW );

				//write_FaceArea("area.ply", shape, faces, params.triVis, params.triAreas);
				//write_FaceShadow("shadow.ply", shape, faces, params.triVis, params.triNoShadow);
				//getchar();
				//double time = (double)cv::getTickCount();
				cCost = updateHessianMatrix(false, alpha,beta,renderParams,faces,colorIm,lmVisInd,landIm,params,exprW,false);
				if (iter == 0) fCost = cCost;
				if (bCost > cCost || iter == 0){
					alpha_bk.release(); beta_bk.release(); 
					alpha_bk = alpha.clone();
					beta_bk = beta.clone();
					exprW_bk.release();
					exprW_bk = exprW.clone();
					bCost = cCost;
					badCount = 0;
					bestIter = iter;
					memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
				}
				else {
					badCount++;
				}
				//getchar();
				//if (badCount > 2) break;
				//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
				//std::cout << "Times passed: " << time << std::endl;
				//time = (double)cv::getTickCount();
				//std::cout << "hess " << params.hessDiag << std::endl;
				//getchar();
			}
			sno_step2(false, alpha, beta, renderParams, faces,colorIm,lmVisInd,landIm,params, exprW);
			//getchar();
			//params.sF -= 0.002;
		}

		//sprintf(text,"tmp_%05d.png",iter);
		//renderFace(text, colorIm,landIm,false,  alpha, beta, faces, renderParams );
		//alpha = alpha_bk.clone();
		//beta = beta_bk.clone();
		//memcpy(renderParams,renderParams2,sizeof(float)*RENDER_PARAMS_COUNT);
		//std::cout << "alpha " << alpha << std::endl;
		//std::cout << "beta " << beta << std::endl;
		updateTriangles(colorIm,faces,false,  alpha, renderParams, params, exprW );

		//double time = (double)cv::getTickCount();
		cCost = updateHessianMatrix(false, alpha,beta,renderParams,faces,colorIm,lmVisInd,landIm,params, exprW);
		if (bCost > cCost){
			alpha_bk.release(); beta_bk.release(); 
			alpha_bk = alpha.clone();
			beta_bk = beta.clone();
			exprW_bk.release();
			exprW_bk = exprW.clone();
			bCost = cCost;
			badCount = 0;
			bestIter = iter;
			memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
		}
	}
	//shape.release(); tex.release();
	//shape = festimator.getShape(alpha);
	//tex = festimator.getTexture(beta);
	//sprintf(text,"%s_99.ply",model_file.c_str());
	//write_plyFloat(text,shape,tex,faces);
	////getchar();
	//sprintf(text,"%s_99.alpha",model_file.c_str());
	//ff=fopen(text,"w");
	//for (int i=0;i<alpha.rows;i++) fprintf(ff,"%f\n",alpha.at<float>(i,0));
	//fclose(ff);
	//sprintf(text,"%s_99.beta",model_file.c_str());
	//ff=fopen(text,"w");
	//for (int i=0;i<beta.rows;i++) fprintf(ff,"%f\n",beta.at<float>(i,0));
	//fclose(ff);

	//alpha0 = cv::repeat(alpha,4,1);
	//beta0 = cv::repeat(beta,4,1);
	//alpha = alpha0;
	//beta = beta0;
	//		params.sI = 5.0f;
	//mstep = 0.002;
	//		params.sF = 8.0f;
	//alpha_bk = alpha.clone();
	//beta_bk = beta.clone();
	//iter = 0;badCount = 0;
	//time = (double)cv::getTickCount();
	//for (;iter<8450;iter++) {
	//	if ((iter-8000)%150 == 0) {
	//		params.sI += 0.5f;
	//		if ( params.sF >= 1.0f)
	//		params.sF -= 1.0f;
	//		sprintf(text,"tmp_%05d.png",iter);
	//		renderFace(text, colorIm,landIm,true,  alpha, beta, faces, renderParams );
	//		updateTriangles(colorIm,faces,true,  alpha, renderParams, params );
	//		cCost = updateHessianMatrix(true, alpha,beta,renderParams,faces,colorIm,lmVisInd,landIm,params);
	//		if (bCost > cCost){
	//			alpha_bk.release(); beta_bk.release(); 
	//			alpha_bk = alpha.clone();
	//			beta_bk = beta.clone();
	//			bCost = cCost;
	//		}
	//		else {
	//			badCount++;
	//		}
	//		if (badCount > 1) break;
	//		std::cout << "hess " << params.hessDiag << std::endl;
	//		prevEF = cEF;
	//		time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
	//		std::cout << "Times passed: " << time << std::endl;
	//		time = (double)cv::getTickCount();
	//		getchar();
	//	}
	//	sno_step2(true, alpha, beta, renderParams, faces,colorIm,lmVisInd,landIm,params);
	//	getchar();
	//	params.sF -= 0.002;
	//}
	//		sprintf(text,"tmp_%05d.png",iter);
	//		renderFace(text, colorIm,landIm,true,  alpha, beta, faces, renderParams );
	//
	//alpha = alpha_bk.clone();
	//beta = beta_bk.clone();
	//getchar();
	//std::cout << "alpha end " << alpha << std::endl;
	//std::cout << "beta end" << beta << std::endl;
	//getchar();
	shape = festimator.getShape(alpha, exprW);
	tex = festimator.getTexture(beta);
	//sprintf(text,"%s",model_file.c_str());
	//write_plyFloat(text,shape,tex,faces);

	sprintf(text,"%s.alpha",model_file.c_str());
	FILE* ff=fopen(text,"w");
	for (int i=0;i<alpha.rows;i++) fprintf(ff,"%f\n",alpha.at<float>(i,0));
	fclose(ff);
	//sprintf(text,"%s.cao",model_file.c_str());
	//ff=fopen(text,"w");
	//for (int i=0;i<M;i++) fprintf(ff,"%f ",params.hessDiag.at<float>(i,0));
	//fprintf(ff,"\n");
	//for (int i=0;i<M;i++) fprintf(ff,"%f ",params.gradVec.at<float>(i,0));
	//fprintf(ff,"\n");
	//fclose(ff);
	sprintf(text,"%s.beta",model_file.c_str());
	ff=fopen(text,"w");
	for (int i=0;i<beta.rows;i++) fprintf(ff,"%f\n",beta.at<float>(i,0));
	fclose(ff);
	sprintf(text,"%s.rend",model_file.c_str());
	ff=fopen(text,"w");
	for (int i=0;i<RENDER_PARAMS_COUNT;i++) fprintf(ff,"%f\n",renderParams[i]);
	fclose(ff);
	sprintf(text,"%s.expr",model_file.c_str());
	ff=fopen(text,"w");
	for (int i=0;i<exprW.rows;i++) fprintf(ff,"%f\n",exprW.at<float>(i,0));
	fclose(ff);
	//float err = eS(alpha, beta, params);
	//sprintf(text,"%s.sym",model_file.c_str());
	//ff=fopen(text,"w");
	//fprintf(ff,"%f\n",err);
	//fclose(ff);

	//sprintf(text,"%s",pose_file.c_str());
	//ff=fopen(text,"w");
	//fprintf(ff,"%f, %f, %f, %f, %f, %f\n",renderParams[0],renderParams[1],renderParams[2], renderParams[3], renderParams[4], renderParams[5]);
	//fclose(ff);
	//write_FaceArea("area.ply", shape, faces, params.triVis, params.triAreas);
	//write_FaceShadow("shadow.ply", shape, faces, params.triVis, params.triNoShadow);

	//cv::Mat refRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	//cv::Mat refDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);

	//bool* visible = new bool[im_render->face_->mesh_.nVertices_];
	//bool* noShadow = new bool[im_render->face_->mesh_.nVertices_];

	//float* r = renderParams + RENDER_PARAMS_R;
	//float* t = renderParams + RENDER_PARAMS_T;
	//im_render->loadModel();
	//im_render->render(r,t,_k[4],refRGB,refDepth);
	//projectCheckVis(im_render, shape, r, t, refDepth, visible);
	////imwrite("pc0.png",refRGB);
	////printf("pose %f %f %f, %f %f %f\n",r[0],r[1],r[2],t[0],t[1],t[2]);

	//cv::Mat trgA(3,1,CV_32F);
	//trgA.at<float>(0,0) = 0.0f;
	//trgA.at<float>(1,0) = 0.0f;
	//trgA.at<float>(2,0) = 1.0f;
	//cv::Mat vecL(3,1,CV_32F);
	//vecL.at<float>(0,0) = cos(renderParams[RENDER_PARAMS_LDIR])*sin(renderParams[RENDER_PARAMS_LDIR+1]);
	//vecL.at<float>(1,0) = sin(renderParams[RENDER_PARAMS_LDIR]);
	//vecL.at<float>(2,0) = cos(renderParams[RENDER_PARAMS_LDIR])*cos(renderParams[RENDER_PARAMS_LDIR+1]);
	//cv::Mat matR = findRotation(vecL,trgA);
	//cv::Mat matR1;
	//cv::Rodrigues(vecR,matR1);
	//cv::Mat matR2;
	//matR2 = matR*matR1;
	//
	//float r2[3];
	//cv::Mat vecR2(3,1,CV_32F,r2);
	//cv::Rodrigues(matR2,vecR2);

	//cv::Mat refRGB2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	//cv::Mat refDepth2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
	//im_render->render(r2,t,_k[4],refRGB2,refDepth2);
	//projectCheckVis(im_render, shape, r2, t, refDepth2, noShadow);
	////imwrite("pcl_0.png",refRGB);
	//
	//rs.estimateColor(shape,tex,faces,visible,noShadow,renderParams,colors);
	//im_render->copyColors(colors);
	//im_render->loadModel();
	//refRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	//refDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
	//im_render->render(r,t,_k[4],refRGB,refDepth);
	////imwrite("pc.png",refRGB);
	//refRGB2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	//refDepth2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
	//im_render->render(r2,t,_k[4],refRGB2,refDepth2);
	////imwrite("pc_l.png",refRGB2);
	////getchar();

	//getchar();
	return true;
}

bool FaceServices2::updateTriangles(cv::Mat colorIm,cv::Mat faces,bool part, cv::Mat alpha, float* renderParams, BFMParams &params, cv::Mat exprW ){
	Mat k_m(3,3,CV_32F,_k);
	//RenderServices rs;
	cv::Mat shape;
	if (!part) shape = festimator.getShape(alpha,exprW);
	else shape = festimator.getShapeParts(alpha,exprW); 

	cv::Mat vecR(3,1,CV_32F), vecT(3,1,CV_32F);
	cv::Mat colors;

	im_render->copyShape(shape);

	cv::Mat refRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	cv::Mat refDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);

	bool* visible = new bool[im_render->face_->mesh_.nVertices_];
	bool* noShadow = new bool[im_render->face_->mesh_.nVertices_];

	float* r = renderParams + RENDER_PARAMS_R;
	float* t = renderParams + RENDER_PARAMS_T;
	for (int i=0;i<3;i++){
		vecR.at<float>(i,0) = r[i];
		vecT.at<float>(i,0) = t[i];
	}
	im_render->loadModel();
	im_render->render(r,t,_k[4],refRGB,refDepth);
	vector<Point2f> projPoints = projectCheckVis2(im_render, shape, r, t, refDepth, visible);

	cv::Mat gradX, gradY, edgeAngles;
	float angleThresh = 2*M_PI/TEXEDGE_ORIENTATION_REG_NUM;
	if (params.sF[FEATURES_TEXTURE_EDGE] > 0 || params.sF[FEATURES_CONTOUR_EDGE] > 0) {
		cv::Mat grayIm, tmpIm;
		cv::cvtColor( refRGB, tmpIm, CV_BGR2GRAY );
		cv::blur( tmpIm, grayIm, cv::Size(3,3) );
		cv::Scharr(grayIm,gradX,CV_32F,1,0);
		cv::Scharr(grayIm,gradY,CV_32F,0,1);
		//cv::imshow("grayIm",grayIm);
		edgeAngles = cv::Mat::zeros(grayIm.rows,grayIm.cols,CV_32F);
	}

	if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
		params.texEdgeVisIndices.clear();
		params.texEdgeVisBin.clear();
		//cv::Mat tmppIm = colorIm.clone();

		for (int i=0;i<texEdgeIndices.size();i++) {
			if (visible[texEdgeIndices[i]])
				params.texEdgeVisIndices.push_back(texEdgeIndices[i]);
		}

		for (int i=0;i<params.texEdgeVisIndices.size();i++) {
			int ix = floor(projPoints.at(params.texEdgeVisIndices[i]).x+0.5);
			int iy = floor(projPoints.at(params.texEdgeVisIndices[i]).y+0.5);
			//cv::circle(tmppIm,cv::Point(ix,iy),1,cv::Scalar(0,0,255));
			if (ix < 0 || iy < 0 || ix > colorIm.cols-1 || iy > colorIm.rows-1)
				params.texEdgeVisBin.push_back(-1);
			else {
				float ange = atan2(gradY.at<float>(iy,ix),gradX.at<float>(iy,ix));
				int bin = floor((ange + M_PI)/angleThresh + 0.5);
				if (bin >= TEXEDGE_ORIENTATION_REG_NUM) bin = bin - TEXEDGE_ORIENTATION_REG_NUM;
				params.texEdgeVisBin.push_back(bin);
			}
		}
		//cv::imwrite("tex.png",tmppIm);
		//getchar();

		//cv::Mat vColors = cv::Mat::zeros(shape.rows,3,CV_32F);
		//for (int i=0;i<params.texEdgeVisIndices.size();i++){
		//	int ind = params.texEdgeVisIndices[i];
		//	vColors.at<float>(ind,0) = vColors.at<float>(ind,1) = vColors.at<float>(ind,2) = 255;
		//}
		//write_plyFloat("edges.ply",shape,vColors,faces);

		//imshow("conMap",refRGB*255);
		//cv::waitKey();
	}

	if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
		params.conEdgeIndices.clear();
		params.conEdgeBin.clear();

		cv::Mat vertexMap, dMap, binMap, tmpIm, conMap;
		vertexMap = -cv::Mat::ones(colorIm.rows,colorIm.cols,CV_32S);
		dMap = 100*cv::Mat::ones(colorIm.rows,colorIm.cols,CV_32F);
		for (int i=0;i<projPoints.size();i++) {
			if (visible[i]){
				int ix = floor(projPoints[i].x+0.5);
				int iy = floor(projPoints[i].y+0.5);
				if (ix >= 0 && iy >= 0 && ix < colorIm.cols && iy < colorIm.rows) {
					float dist = sqrt(pow(projPoints[i].x-ix,2) + pow(projPoints[i].y - iy,2));
					if (dist < dMap.at<float>(iy,ix)){
						vertexMap.at<int>(iy,ix) = i;
						dMap.at<float>(iy,ix) = dist;
					}
				}
			}
		}
		binMap = refDepth < 0.9999;
		binMap.convertTo(binMap,CV_8U);

		Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 3, 3 ), Point( 1, 1 ) );
		erode( binMap, tmpIm, element );
		conMap = binMap - tmpIm;

		for (int i=0;i<conMap.rows;i++) {
			for (int j=0;j<conMap.cols;j++) {
				if (conMap.at<unsigned char>(i,j) > 0 && vertexMap.at<int>(i,j) >= 0 && BaselFace::BaselFace_canContour[vertexMap.at<int>(i,j)]){
					int ind = vertexMap.at<int>(i,j);
					params.conEdgeIndices.push_back(ind);
					float ange = atan2(gradY.at<float>(i,j),gradX.at<float>(i,j));
					int bin = floor((ange + M_PI)/angleThresh + 0.5);
					bin = bin % (TEXEDGE_ORIENTATION_REG_NUM/2);
					params.conEdgeBin.push_back(bin);
				}
			}
		}

		//cv::Mat vColors = cv::Mat::zeros(shape.rows,3,CV_32F);
		//for (int i=0;i<params.conEdgeIndices.size();i++){
		//	int ind = params.conEdgeIndices[i];
		//	vColors.at<float>(ind,0) = vColors.at<float>(ind,1) = vColors.at<float>(ind,2) = 255;
		//}
		//write_plyFloat("contour.ply",shape,vColors,faces);

		//imshow("conMap",conMap*255);
		//cv::waitKey();
	}
	cv::Mat trgA(3,1,CV_32F);
	trgA.at<float>(0,0) = 0.0f;
	trgA.at<float>(1,0) = 0.0f;
	trgA.at<float>(2,0) = 1.0f;
	cv::Mat vecL(3,1,CV_32F);
	vecL.at<float>(0,0) = cos(renderParams[RENDER_PARAMS_LDIR])*sin(renderParams[RENDER_PARAMS_LDIR+1]);
	vecL.at<float>(1,0) = sin(renderParams[RENDER_PARAMS_LDIR]);
	vecL.at<float>(2,0) = cos(renderParams[RENDER_PARAMS_LDIR])*cos(renderParams[RENDER_PARAMS_LDIR+1]);
	cv::Mat matR = findRotation(vecL,trgA);
	cv::Mat matR1;
	cv::Rodrigues(vecR,matR1);
	cv::Mat matR2;
	matR2 = matR*matR1;

	float r2[3];
	float t2[3];
	t2[0] = t2[1] = 0.00001;
	t2[2] = t[2]*1.5;
	cv::Mat vecR2(3,1,CV_32F,r2);
	cv::Rodrigues(matR2,vecR2);

	cv::Mat refRGB2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	cv::Mat refDepth2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
	im_render->render(r2,t2,_k[4],refRGB2,refDepth2);
	projectCheckVis(im_render, shape, r2, t2, refDepth2, noShadow);
	//memcpy(noShadow,visible,sizeof(bool)*im_render->face_->mesh_.nVertices_);

	params.triVis.release(); params.triNoShadow.release();
	params.triVis = cv::Mat::zeros(faces.rows,1,CV_8U);
	params.triNoShadow = cv::Mat::zeros(faces.rows,1,CV_8U);
	for (int i=0;i<faces.rows;i++){
		unsigned int val = 1;
		if (BaselFace::BaselFace_keepV[i] == 0) val = 0;
		else {
			for (int j=0;j<3;j++) {
				if (!visible[faces.at<int>(i,j)]) {
					val = 0; break;
				}
			}
		}
		params.triVis.at<unsigned int>(i,0) = val;
		if (val) {
			for (int j=0;j<3;j++) {
				if (!noShadow[faces.at<int>(i,j)]) {
					val = 0; break;
				}
			}
			params.triNoShadow.at<unsigned int>(i,0) = val;
		}
	}

	params.triAreas.release();
	params.triAreas = cv::Mat::zeros(faces.rows,1,CV_32F);
	float sumArea = 0;
	for (int i=0;i<faces.rows;i++){
		if (params.triVis.at<unsigned int>(i,0)) {
			Point2f a = projPoints[faces.at<int>(i,0)];
			Point2f b = projPoints[faces.at<int>(i,1)];
			Point2f c = projPoints[faces.at<int>(i,2)];
			Point2f v1 = b - a;
			Point2f v2 = c - a;
			params.triAreas.at<float>(i,0) = abs(v1.x*v2.y - v1.y*v2.x);
			sumArea += params.triAreas.at<float>(i,0);
		}
	}

	params.triCSAreas.release();
	params.triCSAreas = cv::Mat::zeros(faces.rows,1,CV_32F);
	float cssum = 0;
	int bin = 0;
	int new_bin;
	int prev_i = 0;
	for (int i=0;i<faces.rows;i++){
		params.triCSAreas.at<float>(i,0) = cssum;
		params.triAreas.at<float>(i,0) /= sumArea;
		cssum += params.triAreas.at<float>(i,0);
		new_bin = floor(cssum * NUM_AREA_BIN);
		if (new_bin > bin) {
			for (int j=bin;j<new_bin;j++)
				params.indexCSArea[j] = prev_i;
			bin = new_bin;
			prev_i = i;
		}
	}
	for (;bin<NUM_AREA_BIN;bin++)
		params.indexCSArea[bin] = prev_i;
	//printf("************ indexing\n");
	//for (int i=0;i<50;i++)
	//	printf("(%d = %d) %f >= %f;  \n",i,params.indexCSArea[i],params.triCSAreas.at<float>(params.indexCSArea[i],0), ((float)i)/NUM_AREA_BIN);
	//getchar();

	shape.release();
	refRGB.release();refDepth.release();
	refRGB2.release();refDepth2.release();
	delete visible; delete noShadow;
	return true;
}
float FaceServices2::updateHessianMatrix(bool part, cv::Mat alpha, cv::Mat beta, float* renderParams, cv::Mat faces, cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat exprW, bool show ){
	int M = alpha.rows;
	int EM = exprW.rows;
	int nTri = 300;
	float step;
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	params.hessDiag.release();
	params.hessDiag = cv::Mat::zeros(2*M+EM+RENDER_PARAMS_COUNT,1,CV_32F);
	//printf("sno_step --------\n"); 
	//std::cout << "alpha " << alpha.t() <<std::endl;
	//std::cout << "beta " << beta.t() <<std::endl;
	//printf("renderParams ");
	//for (int i=0;i<RENDER_PARAMS_COUNT;i++)
	//	printf("(%d) %f, ",i,renderParams[i]);
	//printf("\n");
	cv::Mat alpha2, beta2, expr2;
	cv::Mat centerPoints, centerTexs, normals;
	cv::Mat ccenterPoints, ccenterTexs, cnormals;
	float renderParams2[RENDER_PARAMS_COUNT];

	cv::Mat texEdge3D,conEdge3D,texEdge3D2, conEdge3D2;
	std::vector<cv::Point2f> pPoints;
	cv::Mat rVec(3,1,CV_32F,renderParams+RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F,renderParams+RENDER_PARAMS_T);
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	//double time = (double)cv::getTickCount();
	cES = eS(alpha, beta, params);
	float currEF = eF(part, alpha, lmInds, landIm, renderParams, exprW);
	cEF = currEF;
	cETE = cECE = 0;
	if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
		if (part) texEdge3D = festimator.getTriByAlphaParts(alpha,params.texEdgeVisIndices,exprW);
		else texEdge3D = festimator.getTriByAlpha(alpha,params.texEdgeVisIndices,exprW);
		projectPoints(texEdge3D,rVec,tVec,k_m,distCoef,pPoints);
		cETE = eE(colorIm,pPoints,params,0);
	}
	if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
		if (part) conEdge3D = festimator.getTriByAlphaParts(alpha,params.conEdgeIndices,exprW);
		else conEdge3D = festimator.getTriByAlpha(alpha,params.conEdgeIndices,exprW);
		projectPoints(conEdge3D,rVec,tVec,k_m,distCoef,pPoints);
		cECE = eE(colorIm,pPoints,params,1);
	}
	//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
	//std::cout << "Times passed EF: " << time << std::endl;

	//time = (double)cv::getTickCount();
	std::vector<int> inds;
	randSelectTriangles(nTri, params, inds);
	//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
	//std::cout << "Times passed randSelect: " << time << std::endl;
	//for (int i=0;i< 30;i++)
	//	printf("%d ",inds[i]);
	//printf("\n ");
	//double time = (double)cv::getTickCount();
	if (params.computeEI) {
		getTrianglesCenterNormal(part, alpha, beta,  faces, inds, centerPoints, centerTexs, normals, exprW);
		//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
		//std::cout << "Times passed getTri: " << time << std::endl;
		ccenterPoints = centerPoints.clone();
		ccenterTexs = centerTexs.clone();
		cnormals = normals.clone();
	}
	//time = (double)cv::getTickCount();
	float currEI = eI(colorIm, centerPoints, centerTexs, normals,renderParams,inds,params,show);
	//printf("hesscurrEF %f %f, %f %f\n",currEF,currEI,cETE, cECE);
	//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
	//std::cout << "Times passed EI: " << time << std::endl;
	//write_SelectedTri("tmp_tri.ply", alpha, faces,inds, centerPoints, centerTexs, normals);
	//getchar();
	// alpha
	step = mstep*20;
	//printf("alpha \n");
	if (params.optimizeAB[0]) {
		for (int i=0;i<M; i++){
			alpha2.release(); alpha2 = alpha.clone();
			alpha2.at<float>(i,0) += step;
			float cES2 = eS(alpha2, beta, params);
			float tmpEF1 = eF(part, alpha2, lmInds, landIm, renderParams,exprW);
			//printf("tmpEF1 %f\n",tmpEF1);

			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) texEdge3D2 = festimator.getTriByAlphaParts(alpha2,params.texEdgeVisIndices,exprW);
				else texEdge3D2 = festimator.getTriByAlpha(alpha2,params.texEdgeVisIndices,exprW);
				projectPoints(texEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) conEdge3D2 = festimator.getTriByAlphaParts(alpha2,params.conEdgeIndices,exprW);
				else conEdge3D2 = festimator.getTriByAlpha(alpha2,params.conEdgeIndices,exprW);
				projectPoints(conEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE += cECE2 - cECE;
			}
			if (params.computeEI)
				getTrianglesCenterVNormal(part, alpha2, faces, inds, centerPoints, normals,exprW);
			float tmpEI1 = eI(colorIm, centerPoints, ccenterTexs, normals,renderParams,inds,params);
			//printf("tmpEI1 %f\n",tmpEI1);
			alpha2.at<float>(i,0) -= 2*step;
			float cES3 = eS(alpha2, beta, params);
			float tmpEF2 = eF(part, alpha2, lmInds, landIm, renderParams,exprW);
			//printf("tmpEF2 %f\n",tmpEF2);
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) texEdge3D2 = festimator.getTriByAlphaParts(alpha2,params.texEdgeVisIndices,exprW);
				else texEdge3D2 = festimator.getTriByAlpha(alpha2,params.texEdgeVisIndices,exprW);
				projectPoints(texEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) conEdge3D2 = festimator.getTriByAlphaParts(alpha2,params.conEdgeIndices,exprW);
				else conEdge3D2 = festimator.getTriByAlpha(alpha2,params.conEdgeIndices,exprW);
				projectPoints(conEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE += cECE2 - cECE;
			}

			if (params.computeEI)
				getTrianglesCenterVNormal(part, alpha2,  faces, inds, centerPoints, normals,exprW);
			float tmpEI2 = eI(colorIm, centerPoints, ccenterTexs, normals,renderParams,inds,params);
			//printf("tmpEI2 %f\n",tmpEI2);
			params.hessDiag.at<float>(i,0) = (params.sI * (tmpEI1 - 2*currEI + tmpEI2) + params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2))/(step*step) 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/(step*step) 
				+ 2/(0.25f*M) + cES3+cES2 - 2*cES;
		}
	}
	// beta
	step = mstep*20;
	if (params.optimizeAB[1]) {
		for (int i=0;i<M; i++){
			beta2.release(); beta2 = beta.clone();
			beta2.at<float>(i,0) += step;
			float cES2 = eS(alpha, beta2, params);
			if (params.computeEI)
				getTrianglesCenterTex(part, beta2,  faces, inds, centerTexs);
			float tmpEI1 = eI(colorIm, ccenterPoints, centerTexs, cnormals,renderParams,inds,params);
			//printf("tmpEI1 %f\n",tmpEI1);
			beta2.at<float>(i,0) -= 2*step;
			float cES3 = eS(alpha, beta2, params);
			if (params.computeEI)
				getTrianglesCenterTex(part, beta2,  faces, inds, centerTexs);
			float tmpEI2 = eI(colorIm, ccenterPoints, centerTexs, cnormals,renderParams,inds,params);
			//printf("tmpEI2 %f\n",tmpEI2);
			params.hessDiag.at<float>(M+i,0) = (params.sI * (tmpEI1 - 2*currEI + tmpEI2))/(step*step) + 2/(0.5f*M) + cES3 + cES2 -2*cES;
			//params.hessDiag.at<float>(M+i,0) = 0;
		}
	}
	// expr
	step = mstep*5;
	if (params.optimizeExpr) {
		for (int i=0;i<EM; i++){
			expr2.release(); expr2 = exprW.clone();
			expr2.at<float>(i,0) += step;
			float tmpEF1 = eF(part, alpha, lmInds, landIm, renderParams,expr2);

			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) texEdge3D2 = festimator.getTriByAlphaParts(alpha,params.texEdgeVisIndices,expr2);
				else texEdge3D2 = festimator.getTriByAlpha(alpha,params.texEdgeVisIndices,expr2);
				projectPoints(texEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) conEdge3D2 = festimator.getTriByAlphaParts(alpha,params.conEdgeIndices,expr2);
				else conEdge3D2 = festimator.getTriByAlpha(alpha,params.conEdgeIndices,expr2);
				projectPoints(conEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE += cECE2 - cECE;
			}
			if (params.computeEI)
				getTrianglesCenterVNormal(part, alpha, faces, inds, centerPoints, normals,expr2);
			float tmpEI1 = eI(colorIm, centerPoints, ccenterTexs, normals,renderParams,inds,params);
			//printf("tmpEI1 %f\n",tmpEI1);
			expr2.at<float>(i,0) -= 2*step;
			float tmpEF2 = eF(part, alpha, lmInds, landIm, renderParams,expr2);
			//printf("tmpEF2 %f\n",tmpEF2);
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) texEdge3D2 = festimator.getTriByAlphaParts(alpha,params.texEdgeVisIndices,expr2);
				else texEdge3D2 = festimator.getTriByAlpha(alpha,params.texEdgeVisIndices,expr2);
				projectPoints(texEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) conEdge3D2 = festimator.getTriByAlphaParts(alpha,params.conEdgeIndices,expr2);
				else conEdge3D2 = festimator.getTriByAlpha(alpha,params.conEdgeIndices,expr2);
				projectPoints(conEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE += cECE2 - cECE;
			}

			if (params.computeEI)
				getTrianglesCenterVNormal(part, alpha,  faces, inds, centerPoints, normals,expr2);
			float tmpEI2 = eI(colorIm, centerPoints, ccenterTexs, normals,renderParams,inds,params);
			//printf("tmpEI2 %f\n",tmpEI2);
			params.hessDiag.at<float>(2*M+i,0) = (params.sI * (tmpEI1 - 2*currEI + tmpEI2) + params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2))/(step*step) 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/(step*step) 
				+ params.sExpr * 2/(0.25f*29) ;
		}
	}
	// r
	//step = 0.05;
	step = mstep*2;
	//step = 0.02;
	//step = 0.01;
	if (params.doOptimize[RENDER_PARAMS_R]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_R+i] += step;
			float tmpEF1 = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D,rVec2,tVec,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D,rVec2,tVec,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE += cECE2 - cECE;
			}
			//printf("tmpEF1 %f\n",tmpEF1);
			float tmpEI1 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			//printf("tmpEI1 %f\n",tmpEI1);
			renderParams2[RENDER_PARAMS_R+i] -= 2*step;
			float tmpEF2 = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
			//printf("tmpEF2 %f\n",tmpEF2);
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D,rVec2,tVec,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D,rVec2,tVec,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE += cECE2 - cECE;
			}

			float tmpEI2 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			//printf("tmpEI2 %f\n",tmpEI2);
			params.hessDiag.at<float>(2*M+EM+i,0) = (params.sI * (tmpEI1 - 2*currEI + tmpEI2) + params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2))/(step*step) 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/(step*step) 
				+ 2.0f/params.sR[RENDER_PARAMS_R+i];
		}
	}
	// t
	step = mstep*10;
	//step = 0.05;
	//step = 0.1;
	if (params.doOptimize[RENDER_PARAMS_T]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_T+i] += step;
			float tmpEF1 = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D,rVec,tVec2,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D,rVec,tVec2,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE += cECE2 - cECE;
			}
			float tmpEI1 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			renderParams2[RENDER_PARAMS_T+i] -= 2*step;
			float tmpEF2 = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D,rVec,tVec2,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D,rVec,tVec2,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE += cECE2 - cECE;
			}
			float tmpEI2 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_T+i,0) = (params.sI * (tmpEI1 - 2*currEI + tmpEI2) + params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2))/(step*step) 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/(step*step) 
				+ 2.0f/params.sR[RENDER_PARAMS_T+i];
		}
	}
	// AMBIENT
	step = mstep;
	if (params.doOptimize[RENDER_PARAMS_AMBIENT]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_AMBIENT+i] += step;
			float tmpEI1 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			renderParams2[RENDER_PARAMS_AMBIENT+i] -= 2*step;
			float tmpEI2 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_AMBIENT+i,0)  = (params.sI * (tmpEI1 - 2*currEI + tmpEI2))/(step*step) + 2.0f/params.sR[RENDER_PARAMS_AMBIENT+i];
			//params.hessDiag.at<float>(2*M+RENDER_PARAMS_AMBIENT+i,0)  = 0;
		}
	}
	// DIFFUSE
	//step = 0.02;
	if (params.doOptimize[RENDER_PARAMS_DIFFUSE]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_DIFFUSE+i] += step;
			float tmpEI1 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			renderParams2[RENDER_PARAMS_DIFFUSE+i] -= 2*step;
			float tmpEI2 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_DIFFUSE+i,0)  = (params.sI * (tmpEI1 - 2*currEI + tmpEI2))/(step*step) + 2.0f/params.sR[RENDER_PARAMS_DIFFUSE+i];
			//params.hessDiag.at<float>(2*M+RENDER_PARAMS_DIFFUSE+i,0)  = 0;
		}
	}
	// LDIR
	//step = 0.02;
	step = 0.01;
	if (params.doOptimize[RENDER_PARAMS_LDIR]) {
		for (int i=0;i<2; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_LDIR+i] += step;
			float tmpEI1 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			renderParams2[RENDER_PARAMS_LDIR+i] -= 2*step;
			float tmpEI2 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_LDIR+i,0)  = (params.sI * (tmpEI1 - 2*currEI + tmpEI2))/(step*step) + 2.0f/params.sR[RENDER_PARAMS_LDIR+i];
			//params.hessDiag.at<float>(2*M+RENDER_PARAMS_LDIR+i,0)  = 0;
		}
	}
	// others
	//step = 0.01;
	step = mstep;
	for (int i=RENDER_PARAMS_CONTRAST;i<RENDER_PARAMS_COUNT; i++){
		if (params.doOptimize[i]) {
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[i] += step;
			float tmpEI1 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			renderParams2[i] -= 2*step;
			float tmpEI2 = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			params.hessDiag.at<float>(2*M+EM+i,0)  = (params.sI * (tmpEI1 - 2*currEI + tmpEI2))/(step*step) + 2.0f/params.sR[i];
			//params.hessDiag.at<float>(2*M+i,0)  = 0;
		}
	}
	ccenterPoints.release(); ccenterTexs.release();	cnormals.release();
	return currEI;
}

void FaceServices2::randSelectTriangles(int numPoints , BFMParams &params, std::vector<int> &out){
	out.clear();
	cv::RNG rng(cv::getTickCount());
	for (int i=0;i<numPoints;i++){
		float tmp = rng.uniform(0.0f,1.0f);
		int selected = 0;
		int bin = floor(tmp*NUM_AREA_BIN);
		for (int j=params.indexCSArea[bin];j<params.triAreas.rows;j++){
			if (params.triAreas.at<float>(j,0) > 0){
				if (params.triCSAreas.at<float>(j,0) > tmp) break;
				selected = j;
			}
		}
		//printf("tmp %f %d %d %d %f\n",tmp,bin,params.indexCSArea[bin], selected, params.triCSAreas.at<float>(selected,0));
		out.push_back(selected);
	}
}


void FaceServices2::write_FaceArea(char* fname, cv::Mat shape, cv::Mat faces, cv::Mat &vis, cv::Mat &areas){
	std::ofstream ply2( fname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << shape.rows << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "element face " << faces.rows << std::endl;
	ply2 << "property list uchar int vertex_indices\n";
	ply2 << "property uchar red\n";
	ply2 << "property uchar green\n";
	ply2 << "property uchar blue\n";
	ply2 << "end_header\n";
	for( int i = 0; i < shape.rows ; i++ )
	{
		ply2 << shape.at<float>(i,0) << " " << shape.at<float>(i,1) << " " << shape.at<float>(i,2) <<  std::endl;
	}
	for( int i = 0; i < faces.rows ; i++ )
	{
		ply2 << "3 " << faces.at<int>(i,0) << " " << faces.at<int>(i,1) << " " << faces.at<int>(i,2) << " ";
		if (vis.at<unsigned char>(i,0)){
			float val = floor(areas.at<float>(i,0) * 2550000);
			if (val > 255) val = 255;
			if (val < 0) val = 0;
			int c = (int) val;
			ply2 << c << " " << c << " 0";
		}
		else {
			ply2 << " 0 0 0";
		}
		ply2 << std::endl;
	}
	ply2.close();
}
void FaceServices2::write_FaceShadow(char* fname, cv::Mat shape, cv::Mat faces, cv::Mat &vis, cv::Mat &noShadow){
	std::ofstream ply2( fname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << shape.rows << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "element face " << faces.rows << std::endl;
	ply2 << "property list uchar int vertex_indices\n";
	ply2 << "property uchar red\n";
	ply2 << "property uchar green\n";
	ply2 << "property uchar blue\n";
	ply2 << "end_header\n";
	for( int i = 0; i < shape.rows ; i++ )
	{
		ply2 << shape.at<float>(i,0) << " " << shape.at<float>(i,1) << " " << shape.at<float>(i,2) <<  std::endl;
	}
	for( int i = 0; i < faces.rows ; i++ )
	{
		ply2 << "3 " << faces.at<int>(i,0) << " " << faces.at<int>(i,1) << " " << faces.at<int>(i,2) << " ";
		if (vis.at<unsigned char>(i,0)){
			if (noShadow.at<unsigned char>(i,0)){
				ply2 << " 255 0 0";
			}
			else {
				ply2 << " 255 255 255";
			}
		}
		else {
			ply2 << " 0 0 0";
		}
		ply2 << std::endl;
	}
	ply2.close();
}

FaceServices2::~FaceServices2(void)
{
}

std::vector<int> FaceServices2::verticesFromTriangles(cv::Mat faces,std::vector<int> inds0){
	std::vector<int> inds1;
	for (int i=0;i<inds0.size();i++){
		int ind = inds0[i];
		inds1.push_back(faces.at<int>(ind,0));
		inds1.push_back(faces.at<int>(ind,1));
		inds1.push_back(faces.at<int>(ind,2));
	}
	return inds1;
}

std::vector<int> FaceServices2::aVerticesFromTriangles(cv::Mat faces,std::vector<int> inds0, int vind){
	std::vector<int> inds1;
	for (int i=0;i<inds0.size();i++){
		int ind = inds0[i];
		inds1.push_back(faces.at<int>(ind,vind));
	}
	return inds1;
}

void FaceServices2::getTrianglesCenterNormal(bool part, cv::Mat alpha,cv::Mat beta, cv::Mat faces,std::vector<int> inds0, cv::Mat &centerPoints, cv::Mat &centerTexs, cv::Mat &normals, cv::Mat exprW){
	//printf("getTrianglesCenterNormal\n");
	//double time = (double)cv::getTickCount();
	std::vector<int> inds1 = verticesFromTriangles(faces,inds0);
	std::vector<int> inds2 = aVerticesFromTriangles(faces,inds0);
	//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
	//std::cout << "Times passed vTri: " << time << std::endl;

	cv::Mat vertices, texs; 
	if (!part) {
		vertices = festimator.getTriByAlpha(alpha,inds1,exprW);
		texs = festimator.getTriByBeta(beta,inds2);
	}
	else {
		vertices = festimator.getTriByAlphaParts(alpha,inds1,exprW);
		texs = festimator.getTriByBetaParts(beta,inds2);
	}
	//if (part) {
	//	write_plyFloat("randTri.ply",vertices,texs,cv::Mat());
	//	getchar();
	//}
	//time = (double)cv::getTickCount();
	if (centerPoints.cols == 0) centerPoints = cv::Mat::zeros(inds0.size(),3,CV_32F);
	if (centerTexs.cols == 0) centerTexs = cv::Mat::zeros(inds0.size(),3,CV_32F);
	if (normals.cols == 0) normals = cv::Mat::zeros(inds0.size(),3,CV_32F);
	for (int i=0;i<inds0.size();i++){
		Point3f a(vertices.at<float>(3*i,0),vertices.at<float>(3*i,1),vertices.at<float>(3*i,2));
		Point3f b(vertices.at<float>(3*i+1,0),vertices.at<float>(3*i+1,1),vertices.at<float>(3*i+1,2));
		Point3f c(vertices.at<float>(3*i+2,0),vertices.at<float>(3*i+2,1),vertices.at<float>(3*i+2,2));
		Point3f v1 = b-a;
		Point3f v2 = c-a;
		//Point3f v0 = a+b+c;
		Point3f v = v1.cross(v2);
		float normm = sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
		//centerPoints.at<float>(i,0) = v0.x/3;
		//centerPoints.at<float>(i,1) = v0.y/3;
		//centerPoints.at<float>(i,2) = v0.z/3;
		centerPoints.at<float>(i,0) = a.x;
		centerPoints.at<float>(i,1) = a.y;
		centerPoints.at<float>(i,2) = a.z;
		normals.at<float>(i,0) = v.x/normm;
		normals.at<float>(i,1) = v.y/normm;
		normals.at<float>(i,2) = v.z/normm;
	}
	for (int i=0;i<inds0.size();i++){
		//Point3f a(texs.at<float>(3*i,0),texs.at<float>(3*i,1),texs.at<float>(3*i,2));
		//Point3f b(texs.at<float>(3*i+1,0),texs.at<float>(3*i+1,1),texs.at<float>(3*i+1,2));
		//Point3f c(texs.at<float>(3*i+2,0),texs.at<float>(3*i+2,1),texs.at<float>(3*i+2,2));
		//Point3f v0 = a+b+c;
		centerTexs.at<float>(i,0) = texs.at<float>(i,0);
		centerTexs.at<float>(i,1) = texs.at<float>(i,1);
		centerTexs.at<float>(i,2) = texs.at<float>(i,2);
	}
	//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
	//std::cout << "Times passed compute: " << time << std::endl;
}

void FaceServices2::getTrianglesCenterVNormal(bool part, cv::Mat alpha, cv::Mat faces,std::vector<int> inds0, cv::Mat &centerPoints, cv::Mat &normals, cv::Mat exprW){
	//printf("getTrianglesCenterVNormal\n");
	std::vector<int> inds1 = verticesFromTriangles(faces,inds0);
	cv::Mat vertices; 
	if (!part) {
		vertices = festimator.getTriByAlpha(alpha,inds1,exprW);
	}
	else {
		vertices = festimator.getTriByAlphaParts(alpha,inds1,exprW);
	}
	if (centerPoints.cols == 0) centerPoints = cv::Mat::zeros(inds0.size(),3,CV_32F);
	if (normals.cols == 0) normals = cv::Mat::zeros(inds0.size(),3,CV_32F);
	for (int i=0;i<inds0.size();i++){
		Point3f a(vertices.at<float>(3*i,0),vertices.at<float>(3*i,1),vertices.at<float>(3*i,2));
		Point3f b(vertices.at<float>(3*i+1,0),vertices.at<float>(3*i+1,1),vertices.at<float>(3*i+1,2));
		Point3f c(vertices.at<float>(3*i+2,0),vertices.at<float>(3*i+2,1),vertices.at<float>(3*i+2,2));
		Point3f v1 = b-a;
		Point3f v2 = c-a;
		//Point3f v0 = a+b+c;
		Point3f v = v1.cross(v2);
		float normm = sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
		//centerPoints.at<float>(i,0) = v0.x/3;
		//centerPoints.at<float>(i,1) = v0.y/3;
		//centerPoints.at<float>(i,2) = v0.z/3;
		centerPoints.at<float>(i,0) = a.x;
		centerPoints.at<float>(i,1) = a.y;
		centerPoints.at<float>(i,2) = a.z;
		normals.at<float>(i,0) = v.x/normm;
		normals.at<float>(i,1) = v.y/normm;
		normals.at<float>(i,2) = v.z/normm;
	}
}

void FaceServices2::getTrianglesCenterTex(bool part, cv::Mat beta, cv::Mat faces,std::vector<int> inds0, cv::Mat &centerTexs){
	//printf("getTrianglesCenterTex\n");
	std::vector<int> inds2 = aVerticesFromTriangles(faces,inds0);
	cv::Mat texs; 
	if (!part) {
		texs = festimator.getTriByBeta(beta,inds2);
	}
	else {
		texs = festimator.getTriByBetaParts(beta,inds2);
	}
	if (centerTexs.cols == 0) centerTexs = cv::Mat::zeros(inds0.size(),3,CV_32F);
	for (int i=0;i<inds0.size();i++){
		//Point3f a(texs.at<float>(3*i,0),texs.at<float>(3*i,1),texs.at<float>(3*i,2));
		//Point3f b(texs.at<float>(3*i+1,0),texs.at<float>(3*i+1,1),texs.at<float>(3*i+1,2));
		//Point3f c(texs.at<float>(3*i+2,0),texs.at<float>(3*i+2,1),texs.at<float>(3*i+2,2));
		//Point3f v0 = a+b+c;
		//centerTexs.at<float>(i,0) = v0.x/3;
		//centerTexs.at<float>(i,1) = v0.y/3;
		//centerTexs.at<float>(i,2) = v0.z/3;
		centerTexs.at<float>(i,0) = texs.at<float>(i,0);
		centerTexs.at<float>(i,1) = texs.at<float>(i,1);
		centerTexs.at<float>(i,2) = texs.at<float>(i,2);
	}
}

cv::Mat FaceServices2::computeGradient(bool part, cv::Mat alpha, cv::Mat beta, float* renderParams, cv::Mat faces,cv::Mat colorIm, std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, std::vector<int> &inds, cv::Mat exprW){
	int M = alpha.rows;
	int EM = exprW.rows;
	int nTri = 40;
	float step;
	double time;
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	cv::Mat out(2*M+EM+RENDER_PARAMS_COUNT,1,CV_32F);

	cv::Mat alpha2, beta2, expr2;
	cv::Mat centerPoints, centerTexs, normals;
	cv::Mat ccenterPoints, ccenterTexs, cnormals;
	cv::Mat texEdge3D,conEdge3D,texEdge3D2, conEdge3D2;
	std::vector<cv::Point2f> pPoints;
	float renderParams2[RENDER_PARAMS_COUNT];
	cv::Mat rVec(3,1,CV_32F,renderParams+RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F,renderParams+RENDER_PARAMS_T);
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	cES = eS(alpha, beta, params);
	//printf("%f\n",cES);
	float currEF = eF(part, alpha, lmInds, landIm, renderParams,exprW);
	cETE = cECE = 0;
	if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
		if (part) texEdge3D = festimator.getTriByAlphaParts(alpha,params.texEdgeVisIndices,exprW);
		else texEdge3D = festimator.getTriByAlpha(alpha,params.texEdgeVisIndices,exprW);
		projectPoints(texEdge3D,rVec,tVec,k_m,distCoef,pPoints);
		cETE = eE(colorIm,pPoints,params,0);
	}
	if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
		if (part) conEdge3D = festimator.getTriByAlphaParts(alpha,params.conEdgeIndices,exprW);
		else conEdge3D = festimator.getTriByAlpha(alpha,params.conEdgeIndices,exprW);
		projectPoints(conEdge3D,rVec,tVec,k_m,distCoef,pPoints);
		cECE = eE(colorIm,pPoints,params,1);
	}
	randSelectTriangles(nTri, params, inds);
	if (params.computeEI) {
		getTrianglesCenterNormal(part, alpha, beta,  faces, inds, centerPoints, centerTexs, normals,exprW);

		ccenterPoints = centerPoints.clone();
		ccenterTexs = centerTexs.clone();
		cnormals = normals.clone();
	}
	float currEI = eI(colorIm, centerPoints, centerTexs, normals,renderParams,inds,params);
	cEI = currEI;
	//if (params.computeEI) getchar();
	//printf("currEF %f %f\n",currEF,currEI);
	cEF = currEF;
	//if (cEF > prevEF) printf("grad -> %f vs. ",currEF);
	// alpha
	//int nthreads = omp_get_num_threads();
	//printf("Sequential section: # of threads = %d\n",nthreads);
	step = mstep*20;
	if (params.optimizeAB[0]) {
//		#pragma omp parallel for
		for (int i=0;i<M; i++){
			cv::Mat centerPoints, normals, texEdge3D2, conEdge3D2;
			std::vector<cv::Point2f> pPoints;
			cv::Mat alpha2 = alpha.clone();
			alpha2.at<float>(i,0) += step;
			float tmpEF = eF(part, alpha2, lmInds, landIm, renderParams,exprW);
			//if (i==M-1) 	time = (double)cv::getTickCount();
			if (params.computeEI) getTrianglesCenterVNormal(part, alpha2,  faces, inds, centerPoints, normals,exprW);
			float tmpEI = eI(colorIm, centerPoints, ccenterTexs, normals,renderParams,inds,params);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) texEdge3D2 = festimator.getTriByAlphaParts(alpha2,params.texEdgeVisIndices,exprW);
				else texEdge3D2 = festimator.getTriByAlpha(alpha2,params.texEdgeVisIndices,exprW);
				projectPoints(texEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE = cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) conEdge3D2 = festimator.getTriByAlphaParts(alpha2,params.conEdgeIndices,exprW);
				else conEdge3D2 = festimator.getTriByAlpha(alpha2,params.conEdgeIndices,exprW);
				projectPoints(conEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE = cECE2 - cECE;
			}
			float cES2 = eS(alpha2, beta, params);
			out.at<float>(i,0) = (params.sI * (tmpEI - currEI) + params.sF[FEATURES_LANDMARK] * (tmpEF - currEF))/step
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/step 
				+ 2*alpha.at<float>(i,0)/(0.25f*M) + cES2 - cES;
			//if (cEF > prevEF) printf("%f, ",tmpEF);
		}
	}
	// beta
	step = mstep*20;
	if (params.optimizeAB[1]) {
//		#pragma omp parallel for
		for (int i=0;i<M; i++){
			cv::Mat centerTexs;
			cv::Mat beta2 = beta.clone();
			beta2.at<float>(i,0) += step;
			if (params.computeEI)
				getTrianglesCenterTex(part, beta2,  faces, inds, centerTexs);
			float tmpEI = eI(colorIm, ccenterPoints, centerTexs, cnormals,renderParams,inds,params);
			float cES2 = eS(alpha, beta2, params);
			out.at<float>(M+i,0) = (params.sI * (tmpEI - currEI))/step + 2*beta.at<float>(i,0)/(0.5f*M) + cES2 - cES;
		}
	}
	// expr
	step = mstep*5;
	if (params.optimizeExpr) {
//		#pragma omp parallel for
		for (int i=0;i<EM; i++){
			cv::Mat centerPoints, normals, texEdge3D2, conEdge3D2;
			std::vector<cv::Point2f> pPoints;
			cv::Mat expr2 = exprW.clone();
			expr2.at<float>(i,0) += step;
			float tmpEF = eF(part, alpha, lmInds, landIm, renderParams,expr2);
			//if (i==M-1) 	time = (double)cv::getTickCount();
			if (params.computeEI) getTrianglesCenterVNormal(part, alpha,  faces, inds, centerPoints, normals,expr2);
			float tmpEI = eI(colorIm, centerPoints, ccenterTexs, normals,renderParams,inds,params);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) texEdge3D2 = festimator.getTriByAlphaParts(alpha,params.texEdgeVisIndices,expr2);
				else texEdge3D2 = festimator.getTriByAlpha(alpha,params.texEdgeVisIndices,expr2);
				projectPoints(texEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE = cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) conEdge3D2 = festimator.getTriByAlphaParts(alpha,params.conEdgeIndices,expr2);
				else conEdge3D2 = festimator.getTriByAlpha(alpha,params.conEdgeIndices,expr2);
				projectPoints(conEdge3D2,rVec,tVec,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE = cECE2 - cECE;
			}
			out.at<float>(2*M+i,0) = (params.sI * (tmpEI - currEI) + params.sF[FEATURES_LANDMARK] * (tmpEF - currEF))/step
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/step 
				+ params.sExpr * 2*exprW.at<float>(i,0)/(0.25f*29);
			//if (cEF > prevEF) printf("%f, ",tmpEF);
		}
	}
	// r
	//step = 0.05;
	//step = 0.01;
	step = mstep*2;
	//step = 0.01;
	if (params.doOptimize[RENDER_PARAMS_R]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_R+i] += step;
			float tmpEF = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D,rVec2,tVec,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE = cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D,rVec2,tVec,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE = cECE2 - cECE;
			}
			float tmpEI = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			out.at<float>(2*M+EM+i,0) = (params.sI * (tmpEI - currEI) + params.sF[FEATURES_LANDMARK] * (tmpEF - currEF))/step 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/step 
				+ 2*(renderParams[RENDER_PARAMS_R+i] - params.initR[RENDER_PARAMS_R+i])/params.sR[RENDER_PARAMS_R+i];
			//if (cEF > prevEF) printf("%f, ",tmpEF);
		}
	}
	// t
	step = mstep*10;
	//step = 1;
	//step = 0.1;
	if (params.doOptimize[RENDER_PARAMS_T]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_T+i] += step;
			float tmpEF = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D,rVec,tVec2,k_m,distCoef,pPoints);
				float cETE2 = eE(colorIm,pPoints,params,0);
				dTE = cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D,rVec,tVec2,k_m,distCoef,pPoints);
				float cECE2 = eE(colorIm,pPoints,params,1);
				dCE = cECE2 - cECE;
			}
			float tmpEI = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			out.at<float>(2*M+EM+RENDER_PARAMS_T+i,0) = (params.sI * (tmpEI - currEI) + params.sF[FEATURES_LANDMARK] * (tmpEF - currEF))/step 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/step 
				+ 2*(renderParams[RENDER_PARAMS_T+i] - params.initR[RENDER_PARAMS_T+i])/params.sR[RENDER_PARAMS_T+i];
			//if (cEF > prevEF) printf("%f, ",tmpEF);
		}
	}
	// AMBIENT
	step = mstep;
	if (params.doOptimize[RENDER_PARAMS_AMBIENT]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_AMBIENT+i] += step;
			float tmpEI = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			out.at<float>(2*M+EM+RENDER_PARAMS_AMBIENT+i,0) = (params.sI * (tmpEI - currEI))/step + 2*(renderParams[RENDER_PARAMS_AMBIENT+i] - params.initR[RENDER_PARAMS_AMBIENT+i])/params.sR[RENDER_PARAMS_AMBIENT+i];
			//out.at<float>(2*M+RENDER_PARAMS_AMBIENT+i,0) = 0;
		}
	}
	// DIFFUSE
	//step = 0.02;
	if (params.doOptimize[RENDER_PARAMS_DIFFUSE]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_DIFFUSE+i] += step;
			float tmpEI = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			out.at<float>(2*M+EM+RENDER_PARAMS_DIFFUSE+i,0) = (params.sI * (tmpEI - currEI))/step + 2*(renderParams[RENDER_PARAMS_DIFFUSE+i] - params.initR[RENDER_PARAMS_DIFFUSE+i])/params.sR[RENDER_PARAMS_DIFFUSE+i];
			//out.at<float>(2*M+RENDER_PARAMS_DIFFUSE+i,0) = 0;
		}
	}
	// LDIR
	//step = 0.02;
	step = 0.01;
	if (params.doOptimize[RENDER_PARAMS_LDIR]) {
		for (int i=0;i<2; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_LDIR+i] += step;
			float tmpEI = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			out.at<float>(2*M+EM+RENDER_PARAMS_LDIR+i,0) = (params.sI * (tmpEI - currEI))/step + 2*(renderParams[RENDER_PARAMS_LDIR+i] - params.initR[RENDER_PARAMS_LDIR+i])/params.sR[RENDER_PARAMS_LDIR+i];
			//out.at<float>(2*M+RENDER_PARAMS_LDIR+i,0) = 0;
		}
	}
	// others
	//step = 0.01;
	step = mstep;
	for (int i=RENDER_PARAMS_CONTRAST;i<RENDER_PARAMS_COUNT; i++){
		//out.at<float>(2*M+i,0) = 0;
		if (params.doOptimize[i]) {
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[i] += step;
			float tmpEI = eI(colorIm, ccenterPoints, ccenterTexs, cnormals,renderParams2,inds,params);
			out.at<float>(2*M+EM+i,0) = (params.sI * (tmpEI - currEI))/step + 2*(renderParams[i] - params.initR[i])/params.sR[i];
		}
	}
	ccenterPoints.release(); ccenterTexs.release(); cnormals.release();
	return out;
}

float FaceServices2::eF(bool part, cv::Mat alpha, std::vector<int> inds, cv::Mat landIm, float* renderParams, cv::Mat exprW){
	Mat k_m(3,3,CV_32F,_k);
	//printf("%f\n",renderParams[RENDER_PARAMS_R+1]);
	cv::Mat mLM;
	if (!part)
		mLM = festimator.getLMByAlpha(alpha,-renderParams[RENDER_PARAMS_R+1], inds, exprW);
	else
		mLM = festimator.getLMByAlphaParts(alpha,-renderParams[RENDER_PARAMS_R+1], inds, exprW);
	//if (part) {
	//	write_plyFloat("vismLM.ply",mLM.t());
	//	getchar();
	//}
	cv::Mat rVec(3,1,CV_32F, renderParams + RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F, renderParams + RENDER_PARAMS_T);
	std::vector<cv::Point2f> allImgPts;
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );

	cv::projectPoints( mLM, rVec, tVec, k_m, distCoef, allImgPts );
	float err = 0;
	for (int i=0;i<mLM.rows;i++){
		float val = landIm.at<float>(i,0) - allImgPts[i].x;
		err += val*val;
		val = landIm.at<float>(i,1) - allImgPts[i].y;
		err += val*val;
	}
	return sqrt(err/mLM.rows);
}

float FaceServices2::eI(cv::Mat colorIm,cv::Mat &centerPoints, cv::Mat &centerTexs, cv::Mat &normals, float* renderParams,std::vector<int> inds, BFMParams &params, bool show){
	if (!params.computeEI) return 0;
	Mat k_m(3,3,CV_32F,_k);
	cv::Mat colors;
	//RenderServices rs;
	rs.estimatePointColor(centerPoints, centerTexs, normals, inds, params.triNoShadow, renderParams, colors);

	cv::Mat rVec(3,1,CV_32F, renderParams + RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F, renderParams + RENDER_PARAMS_T);
	std::vector<cv::Point2f> allImgPts;
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );

	//printf("projectPoints\n");
	cv::projectPoints( centerPoints, rVec, tVec, k_m, distCoef, allImgPts );
	float err = 0;
	cv::Mat colorIm2;
	if (show) colorIm2 = colorIm.clone();
	for (int i=0;i<centerPoints.rows;i++){
		//if (allImgPts[i].x < 0 || allImgPts[i].x >= colorIm.cols-1 || allImgPts[i].y < 0 || allImgPts[i].y >= colorIm.rows-1)
		//	printf("outer point\n");
                CvPoint2D64f tmpPt = cvPoint2D64f(allImgPts[i].x,allImgPts[i].y);
		cv::Vec3d imcolor = avSubMatValue8UC3_2( &tmpPt, &colorIm );
		if (show) cv::circle(colorIm2,cv::Point(allImgPts[i].x,allImgPts[i].y),1,cv::Scalar(colors.at<float>(i,2),colors.at<float>(i,1),colors.at<float>(i,0)),2);
		//cv::Vec3b imcolor = colorIm.at<Vec3b>(allImgPts[i].y,allImgPts[i].x);
		for (int j=0;j<3;j++){
			float val = (colors.at<float>(i,j) - imcolor(2-j))/5.0f;
			if (abs(val) < 4) 
				err += val*val;
			else
				err += 2*4*abs(val) - 4*4;
		}
	}
	if (show) cv::imwrite("pp.png",colorIm2);
	return sqrt(err/centerPoints.rows);
}

void FaceServices2::sno_step(bool part, cv::Mat &alpha, cv::Mat &beta, float* renderParams, cv::Mat faces,cv::Mat colorIm, std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW){
	//printf("sno_step --------\n"); 
	//std::cout << "alpha " << alpha.t() <<std::endl;
	//std::cout << "beta " << beta.t() <<std::endl;
	//printf("renderParams ");
	//for (int i=0;i<RENDER_PARAMS_COUNT;i++)
	//	printf("%f, ",renderParams[i]);
	//printf("\n");
	float lambda = 0.005;
	std::vector<int> inds;
	cv::Mat dE = computeGradient(part, alpha, beta, renderParams, faces, colorIm, lmInds, landIm, params, inds,exprW);
	//std::cout << "**dE " << dE.t() <<std::endl;
	//if (cEF > prevEF) printf("EF increase: (%f -> %f) with step (",prevEF,cEF);
	int M = alpha.rows;
	int EM = exprW.rows;
	if (params.optimizeAB[0]){
		for (int i=0;i<M;i++)
			if (abs(params.hessDiag.at<float>(i,0)) > 0.0000001) {
				//if (params.hessDiag.at<float>(i,0) > 0.0000001) {
				float change = - lambda*dE.at<float>(i,0)/abs(params.hessDiag.at<float>(i,0));
				if (change > mstep/5) change = mstep/5;
				if (change < -mstep/5) change = -mstep/5;
				if (alpha.at<float>(i,0) + change > 4.0) change = (4-alpha.at<float>(i,0))/10;
				if (alpha.at<float>(i,0) + change < -4.0) change = (-4 - alpha.at<float>(i,0))/10;
				alpha.at<float>(i,0) += change;
				//if (cEF > prevEF) printf("%f, ",change);
			}
			//if (cEF > prevEF) printf(") (");
	}

	if (params.optimizeAB[1]){
		for (int i=0;i<M;i++)
			if (abs(params.hessDiag.at<float>(M+i,0)) > 0.0000001) {
				//if (params.hessDiag.at<float>(M+i,0) > 0.0000001) {
				float change = - lambda*dE.at<float>(M+i,0)/abs(params.hessDiag.at<float>(M+i,0));
				if (change > mstep/2) change = mstep/2;
				if (change < -mstep/2) change = -mstep/2;
				if (beta.at<float>(i,0) + change > 4.0) change = (4-beta.at<float>(i,0))/10;
				if (beta.at<float>(i,0) + change < -4.0) change = (-4 - beta.at<float>(i,0))/10;
				beta.at<float>(i,0) += change;
			}
	}
	
	if (params.optimizeExpr){
		for (int i=0;i<EM;i++)
			if (abs(params.hessDiag.at<float>(i+2*M,0)) > 0.0000001) {
				//if (params.hessDiag.at<float>(i,0) > 0.0000001) {
				float change = - lambda*dE.at<float>(i+2*M,0)/abs(params.hessDiag.at<float>(i+2*M,0));
				if (change > mstep/5) change = mstep/5;
				if (change < -mstep/5) change = -mstep/5;
				if (exprW.at<float>(i,0) + change > 3.0) change = (3-exprW.at<float>(i,0))/10;
				if (exprW.at<float>(i,0) + change < -3.0) change = (-3 - exprW.at<float>(i,0))/10;
				exprW.at<float>(i,0) += change;
				//if (cEF > prevEF) printf("%f, ",change);
			}
			//if (cEF > prevEF) printf(") (");
	}
	for (int i=0;i<RENDER_PARAMS_COUNT;i++)
		if (params.doOptimize[i]){
			if (abs(params.hessDiag.at<float>(2*M+EM+i,0)) > 0.0000001) {
				//if (params.hessDiag.at<float>(2*M+i,0) > 0.0000001){
				float change = - lambda*dE.at<float>(2*M+EM+i,0)/abs(params.hessDiag.at<float>(2*M+EM+i,0));
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (change > mstep/5) change = mstep/5;
					if (change < -mstep/5) change = -mstep/5;
					if (renderParams[i] + change > 1.0) change = 1-renderParams[i];
					if (renderParams[i] + change < 0) change = -renderParams[i];

				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (change > mstep/10) change = mstep/10;
					if (change < -mstep/10) change = -mstep/10;
					if (renderParams[i] + change > 3.0) change = 3-renderParams[i];
					if (renderParams[i] + change < 0) change = -renderParams[i];
				}
				else if (i < 3 || i>= 6) {
					if (change > mstep/10) change = mstep/10;
					if (change < -mstep/10) change = -mstep/10;
				}
				else {
					if (change > mstep*100) change = mstep*100;
					if (change < -mstep*100) change = -mstep*100;
				}
				renderParams[i] += change;
				//if (cEF > prevEF) printf("%f, ",change);
			}
		}

		//if (cEF > prevEF) {
		//	printf(")");
		//	getchar();
		//}
		prevEF = cEF;
}

void FaceServices2::sno_step2(bool part, cv::Mat &alpha, cv::Mat &beta, float* renderParams, cv::Mat faces,cv::Mat colorIm, std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW){
	//printf("sno_step --------\n"); 
	//std::cout << "alpha " << alpha.t() <<std::endl;
	//std::cout << "beta " << beta.t() <<std::endl;
	//printf("renderParams ");
	//for (int i=0;i<RENDER_PARAMS_COUNT;i++)
	//	printf("%f, ",renderParams[i]);
	//printf("\n");
	float lambda = 0.005;
	std::vector<int> inds;
	cv::Mat dE = computeGradient(part, alpha, beta, renderParams, faces, colorIm, lmInds, landIm, params,inds, exprW);
	params.gradVec.release(); params.gradVec = dE.clone();
	cv::Mat dirMove = dE*0;

	int M = alpha.rows;
	int EM = exprW.rows;
	if (params.optimizeAB[0]){
		for (int i=0;i<M;i++)
			if (abs(params.hessDiag.at<float>(i,0)) > 0.0000001) {
				dirMove.at<float>(i,0) = - lambda*dE.at<float>(i,0)/abs(params.hessDiag.at<float>(i,0));
			}
	}
	if (params.optimizeAB[1]){
		for (int i=0;i<M;i++)
			if (abs(params.hessDiag.at<float>(M+i,0)) > 0.0000001) {
				dirMove.at<float>(M+i,0) = - lambda*dE.at<float>(M+i,0)/abs(params.hessDiag.at<float>(M+i,0));
			}
	}
	if (params.optimizeExpr){
		for (int i=0;i<EM;i++)
			if (abs(params.hessDiag.at<float>(2*M+i,0)) > 0.0000001) {
				dirMove.at<float>(2*M+i,0) = - lambda*dE.at<float>(2*M+i,0)/abs(params.hessDiag.at<float>(2*M+i,0));
			}
	}

	for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
		if (params.doOptimize[i]){
			if (abs(params.hessDiag.at<float>(2*M+EM+i,0)) > 0.0000001) {
				dirMove.at<float>(2*M+EM+i,0) = - lambda*dE.at<float>(2*M+EM+i,0)/abs(params.hessDiag.at<float>(2*M+EM+i,0));
			}
		}
	}

	float pc = line_search(part, alpha, beta, renderParams, dirMove,inds, faces, colorIm, lmInds, landIm, params, exprW, 20);
	//printf("pc = %f\n",pc);
	if (pc == 0) countFail++;
	else {
		if (params.optimizeAB[0]){
			for (int i=0;i<M;i++) {
				alpha.at<float>(i,0) += pc*dirMove.at<float>(i,0);
				if (alpha.at<float>(i,0) > maxVal) alpha.at<float>(i,0) = maxVal;
				else if (alpha.at<float>(i,0) < -maxVal) alpha.at<float>(i,0) = -maxVal;
			}
		}
		if (params.optimizeAB[1]){
			for (int i=0;i<M;i++) {
				beta.at<float>(i,0) += pc*dirMove.at<float>(M+i,0);
				if (beta.at<float>(i,0) > maxVal) beta.at<float>(i,0) = maxVal;
				else if (beta.at<float>(i,0) < -maxVal) beta.at<float>(i,0) = -maxVal;
			}
		}
		if (params.optimizeExpr){
			for (int i=0;i<EM;i++) {
				exprW.at<float>(i,0) += pc*dirMove.at<float>(i+2*M,0);
				if (exprW.at<float>(i,0) > 3) exprW.at<float>(i,0) = 3;
				else if (exprW.at<float>(i,0) < -3) exprW.at<float>(i,0) = -3;
			}
		}

		for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
			if (params.doOptimize[i]){
				renderParams[i] += pc*dirMove.at<float>(2*M+EM+i,0);
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (renderParams[i] > 1.0) renderParams[i] = 1.0;
					if (renderParams[i] < 0) renderParams[i] = 0;

				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (renderParams[i] > 3.0) renderParams[i]  = 3;
					if (renderParams[i] < 0.3) renderParams[i] = 0.3;
				}
			}
		}
	}
	prevEF = cEF;
}

float FaceServices2::line_search(bool part, cv::Mat &alpha, cv::Mat &beta, float* renderParams, cv::Mat &dirMove,std::vector<int> inds, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW, int maxIters){
	float step = 1.0f;
	float sstep = 2.0f;
	float minStep = 0.00001f;
	//float cCost = computeCost(cEF, cEI,cETE, cECE, alpha, beta, renderParams,params);
	cv::Mat alpha2, beta2, exprW2;
	float renderParams2[RENDER_PARAMS_COUNT];
	alpha2 = alpha.clone();
	beta2 = beta.clone();
	exprW2 = exprW.clone();
	memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);

	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	cv::Mat texEdge3D2, conEdge3D2;
	std::vector<cv::Point2f> pPoints;
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	int M = alpha.rows;
	int EM = exprW.rows;
	float ssize = 0;
	for (int i=0;i<dirMove.rows;i++) ssize += dirMove.at<float>(i,0)*dirMove.at<float>(i,0);
	ssize = sqrt(ssize);
	//printf("ssize: %f\n",ssize);
	if (ssize > (2*M+EM+RENDER_PARAMS_COUNT)/160.0f) {
		step = (2*M+EM+RENDER_PARAMS_COUNT)/(160.0f * ssize);
		ssize = (2*M+EM+RENDER_PARAMS_COUNT)/160.0f;
	}
	if (ssize < minStep){
		return 0;
	}
	int tstep = floor(log(ssize/minStep));
	if (tstep < maxIters) maxIters = tstep;

	cv::Mat centerPoints, centerTexs, normals;
	float curCost = computeCost(cEF, cEI,cETE, cECE,cES, alpha, beta, renderParams, params, exprW );
	//printf("curCost %f\n",curCost);

	bool hasNoBound = false;
	int iter = 0;
	for (; iter<maxIters; iter++){
		// update
		if (params.optimizeAB[0]){
			for (int i=0;i<M;i++) {
				float tmp = alpha.at<float>(i,0) + step*dirMove.at<float>(i,0);
				if (tmp >= maxVal) alpha2.at<float>(i,0) = maxVal;
				else if (tmp <= -maxVal) alpha2.at<float>(i,0) = -maxVal;
				else {
					alpha2.at<float>(i,0) = tmp;
					hasNoBound = true;
				}
			}
		}
		if (params.optimizeAB[1]){
			for (int i=0;i<M;i++) {
				float tmp = beta.at<float>(i,0) + step*dirMove.at<float>(M+i,0);
				if (tmp >= maxVal) beta2.at<float>(i,0) = maxVal;
				else if (tmp <= -maxVal) beta2.at<float>(i,0) = -maxVal;
				else {
					beta2.at<float>(i,0) = tmp;
					hasNoBound = true;
				}
			}
		}
		if (params.optimizeExpr){
			for (int i=0;i<EM;i++) {
				float tmp = exprW.at<float>(i,0) + step*dirMove.at<float>(2*M+i,0);
				if (tmp >= 3) exprW2.at<float>(i,0) = 3;
				else if (tmp <= -3) exprW2.at<float>(i,0) = -3;
				else {
					exprW2.at<float>(i,0) = tmp;
					hasNoBound = true;
				}
			}
		}

		for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
			if (params.doOptimize[i]){
				float tmp = renderParams[i] + step*dirMove.at<float>(2*M+EM+i,0);
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (tmp > 1.0) renderParams2[i] = 1.0f;
					else if (tmp < -1.0) renderParams2[i] = -1.0f;
					else {
						renderParams2[i] = tmp;
						hasNoBound = true;
					}
				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (tmp >= 3.0) renderParams2[i] = 3.0f;
					else if (tmp <= -3.0) renderParams2[i] = -3.0f;
					else {
						renderParams2[i] = tmp;
						hasNoBound = true;
					}
				}
				else renderParams2[i] = tmp;
			}
		}
		if (!hasNoBound) {
			iter = maxIters; break;
		}
		float tmpEF = cEF;
		if (params.sF[FEATURES_LANDMARK] > 0) tmpEF = eF(part,alpha2, lmInds,landIm,renderParams2, exprW2);
		float tmpEI = cEI;
		if (params.sI > 0 && params.computeEI) {
			getTrianglesCenterNormal(part, alpha2,beta2,  faces, inds, centerPoints,centerTexs, normals, exprW2);
			tmpEI = eI(colorIm,centerPoints,centerTexs, normals,renderParams2,inds,params);
		}
		float tmpETE = cETE;
		if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
			if (part) texEdge3D2 = festimator.getTriByAlphaParts(alpha2,params.texEdgeVisIndices, exprW2);
			else texEdge3D2 = festimator.getTriByAlpha(alpha2,params.texEdgeVisIndices,exprW2);
			projectPoints(texEdge3D2,rVec2,tVec2,k_m,distCoef,pPoints);
			tmpETE = eE(colorIm,pPoints,params,0);
		}
		float tmpECE = cECE;
		if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
			if (part) conEdge3D2 = festimator.getTriByAlphaParts(alpha2,params.conEdgeIndices,exprW2);
			else conEdge3D2 = festimator.getTriByAlpha(alpha2,params.conEdgeIndices,exprW2);
			projectPoints(conEdge3D2,rVec2,tVec2,k_m,distCoef,pPoints);
			float tmpECE = eE(colorIm,pPoints,params,1);
		}
		
		float tmpES = eS(alpha2, beta2, params);
		float tmpCost = computeCost(tmpEF, tmpEI,tmpETE, tmpECE, tmpES, alpha2, beta2, renderParams2, params,exprW2 );
		//printf("tmpCost %f\n",tmpCost);
		if (tmpCost < curCost) {
			break;
		}
		else {
			step = step/sstep;
			//printf("step %f\n",step);
		}
	}
	//getchar();
	if (iter >= maxIters) return 0;
	else return step;
}

float FaceServices2::computeCost(float vEF, float vEI,float vETE, float vECE, float vS, cv::Mat &alpha, cv::Mat &beta, float* renderParams, BFMParams &params, cv::Mat &exprW ){
	float val = params.sF[FEATURES_LANDMARK]*vEF + params.sI*vEI + params.sF[FEATURES_TEXTURE_EDGE]*vETE + params.sF[FEATURES_CONTOUR_EDGE]*vECE + vS;
	int M = alpha.rows;
	if (params.optimizeAB[0]){
		for (int i=0;i<M;i++)
			val += alpha.at<float>(i,0)*alpha.at<float>(i,0)/(0.25f*M);
	}
	if (params.optimizeAB[1]){
		for (int i=0;i<M;i++)
			val += beta.at<float>(i,0)*beta.at<float>(i,0)/(0.5f*M);
	}
	if (params.optimizeExpr){
		for (int i=0;i<exprW.rows;i++)
			val += params.sExpr * exprW.at<float>(i,0)*exprW.at<float>(i,0)/(0.5f*29);
	}

	for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
		if (params.doOptimize[i]){
			val += (renderParams[i] - params.initR[i])*(renderParams[i] - params.initR[i])/params.sR[i];
		}
	}
	return val;
}

void FaceServices2::renderFace(char* fname, cv::Mat colorIm, cv::Mat landIm, bool part, cv::Mat alpha, cv::Mat beta,cv::Mat faces, float* renderParams, cv::Mat exprW ){
	Mat k_m(3,3,CV_32F,_k);
	//RenderServices rs;
	cv::Mat shape, tex;
	if (!part) {
		shape = festimator.getShape(alpha,exprW);
		tex = festimator.getTexture(beta);
	}
	else {
		shape = festimator.getShapeParts(alpha,exprW);
		tex = festimator.getTextureParts(beta);
	}

	cv::Mat vecR(3,1,CV_32F), vecT(3,1,CV_32F);
	cv::Mat colors;

	im_render->copyShape(shape);

	cv::Mat refRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	cv::Mat refDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);

	bool* visible = new bool[im_render->face_->mesh_.nVertices_];
	bool* noShadow = new bool[im_render->face_->mesh_.nVertices_];

	float* r = renderParams + RENDER_PARAMS_R;
	float* t = renderParams + RENDER_PARAMS_T;
	for (int i=0;i<3;i++){
		vecR.at<float>(i,0) = r[i];
		vecT.at<float>(i,0) = t[i];
	}
	im_render->loadModel();
	im_render->render(r,t,_k[4],refRGB,refDepth);
	vector<Point2f> projPoints = projectCheckVis2(im_render, shape, r, t, refDepth, visible);

	cv::Mat trgA(3,1,CV_32F);
	trgA.at<float>(0,0) = 0.0f;
	trgA.at<float>(1,0) = 0.0f;
	trgA.at<float>(2,0) = 1.0f;
	cv::Mat vecL(3,1,CV_32F);
	vecL.at<float>(0,0) = cos(renderParams[RENDER_PARAMS_LDIR])*sin(renderParams[RENDER_PARAMS_LDIR+1]);
	vecL.at<float>(1,0) = sin(renderParams[RENDER_PARAMS_LDIR]);
	vecL.at<float>(2,0) = cos(renderParams[RENDER_PARAMS_LDIR])*cos(renderParams[RENDER_PARAMS_LDIR+1]);
	cv::Mat matR = findRotation(vecL,trgA);
	cv::Mat matR1;
	cv::Rodrigues(vecR,matR1);
	cv::Mat matR2;
	matR2 = matR*matR1;

	float r2[3];
	float t2[3];
	t2[0] = t2[1] = 0.00001;
	t2[2] = t[2]*1.5;
	cv::Mat vecR2(3,1,CV_32F,r2);
	cv::Rodrigues(matR2,vecR2);

	cv::Mat refRGB2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	cv::Mat refDepth2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
	im_render->render(r2,t2,_k[4],refRGB2,refDepth2);
	projectCheckVis(im_render, shape, r2, t2, refDepth2, noShadow);

	rs.estimateColor(shape,tex,faces,visible,noShadow,renderParams,colors);
	im_render->copyColors(colors);
	im_render->loadModel();
	refRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	refDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
	im_render->render(r,t,_k[4],refRGB,refDepth);
	//for (int i=0;i<landIm.rows;i++){
	//	cv::circle(refRGB,Point(landIm.at<float>(i,0),landIm.at<float>(i,1)),1,Scalar(255,0,0),1);
	//}
	imwrite(fname,refRGB);
	//im_render->render(r2,t2,_k[4],refRGB2,refDepth2);
	//char fname2[200];
	//sprintf(fname2,"l_%s",fname);
	//imwrite(fname2,refRGB2);

	shape.release(); tex.release(); refRGB.release(); refDepth.release();
	refRGB2.release(); refDepth2.release();
	delete visible; delete noShadow;
}

bool FaceServices2::loadReference(string refDir, string model_file, cv::Mat &alpha, cv::Mat &beta, float* renderParams, int &M, cv::Mat &exprW, int &EM){
	string fname(model_file);
	size_t sep = model_file.find_last_of("\\/");
	if (sep != std::string::npos)
		fname = model_file.substr(sep + 1, model_file.size() - sep - 1);
	fname = refDir + fname;

	alpha = cv::Mat::zeros(M,1,CV_32F);
	beta = cv::Mat::zeros(M,1,CV_32F);
	char fpath[250];
	char text[250];
	sprintf(fpath,"%s.alpha",fname.c_str());
	//printf("fpath %s\n",fpath);
	FILE* f = fopen(fpath,"r");
	if (f == 0) return false;

	text[0] = '\0';
	for (int i=0;i<M;i++){
		fgets(text,250,f);
		int l = strlen(text);
		if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
		alpha.at<float>(i,0) = atof(text);
	}
	fclose(f);

	sprintf(fpath,"%s.beta",fname.c_str());
	f = fopen(fpath,"r");
	text[0] = '\0';
	for (int i=0;i<M;i++){
		fgets(text,250,f);
		int l = strlen(text);
		if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
		beta.at<float>(i,0) = atof(text);
	}
	fclose(f);

	sprintf(fpath,"%s.rend",fname.c_str());
	f = fopen(fpath,"r");
	text[0] = '\0';
	for (int i=0;i<RENDER_PARAMS_COUNT;i++){
		fgets(text,250,f);
		int l = strlen(text);
		if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
		renderParams[i] = atof(text);
	}
	fclose(f);
	
	sprintf(fpath,"%s.expr",fname.c_str());
	//printf("fpath %s\n",fpath);
	f = fopen(fpath,"r");
	if (f == 0) return false;

	text[0] = '\0';
	for (int i=0;i<EM;i++){
		fgets(text,250,f);
		int l = strlen(text);
		if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
		exprW.at<float>(i,0) = atof(text);
	}
	fclose(f);
	return true;
}

bool FaceServices2::prepareEdgeDistanceMaps(cv::Mat colorIm){
	if (texEdge == 0) texEdge = new Mat[TEXEDGE_ORIENTATION_REG_NUM];
	if (conEdge == 0) conEdge = new Mat[TEXEDGE_ORIENTATION_REG_NUM/2];
	if (texEdgeDist == 0) texEdgeDist = new Mat[TEXEDGE_ORIENTATION_REG_NUM];
	if (conEdgeDist == 0) conEdgeDist = new Mat[TEXEDGE_ORIENTATION_REG_NUM/2];
	//if (texEdgeDistDX == 0) texEdgeDistDX = new Mat[TEXEDGE_ORIENTATION_REG_NUM];
	//if (texEdgeDistDY == 0) texEdgeDistDY = new Mat[TEXEDGE_ORIENTATION_REG_NUM];
	//if (conEdgeDistDX == 0) conEdgeDistDX = new Mat[TEXEDGE_ORIENTATION_REG_NUM/2];
	//if (conEdgeDistDY == 0) conEdgeDistDY = new Mat[TEXEDGE_ORIENTATION_REG_NUM/2];

	for (int i=0;i<TEXEDGE_ORIENTATION_REG_NUM;i++) 
		texEdge[i] = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8U);
	for (int i=0;i<TEXEDGE_ORIENTATION_REG_NUM/2;i++) 
		conEdge[i] = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8U);

	float threshDist = 10;
	int edgeThresh = 1;
	int lowThreshold = 85;
	int ratio = 25;
	//int minSize = colorIm.rows;
	//if (colorIm.cols < minSize) minSize = colorIm.cols;
	int kernel_size = 5;
	//imwrite("query.png",colorIm);
	//printf("pose %f\n",poseCLM(4));
	cv::Mat grayIm, edgeIm, tmpIm;
	cv::cvtColor( colorIm, tmpIm, CV_BGR2GRAY );
	cv::medianBlur( tmpIm, grayIm, 3 );
	cv::Canny( tmpIm, edgeIm, lowThreshold, lowThreshold*ratio, kernel_size );
	//cv::imwrite("edges.png",edgeIm);

	cv::Mat gradX, gradY;
	cv::Scharr(grayIm,gradX,CV_32F,1,0);
	cv::Scharr(grayIm,gradY,CV_32F,0,1);

	edgeAngles = cv::Mat::zeros(edgeIm.rows,edgeIm.cols,CV_32F);
	float angleThresh = 2*M_PI/TEXEDGE_ORIENTATION_REG_NUM;
	for (int i=0;i<edgeIm.rows;i++){
		for (int j=0;j<edgeIm.cols;j++){
			unsigned char edgeVal = edgeIm.at<unsigned char>(i,j);
			if (edgeVal > 0) {
				float ange = atan2(gradY.at<float>(i,j),gradX.at<float>(i,j));
				edgeAngles.at<float>(i,j) = ange;
				for (int k = 0;k<TEXEDGE_ORIENTATION_REG_NUM;k++) {
					//if (abs(ange - TexAngleCenter[k]) < angleThresh || abs(ange - TexAngleCenter[k] + 2*M_PI) < angleThresh ||  abs(ange - TexAngleCenter[k] - 2*M_PI) < angleThresh) {
						texEdge[k].at<unsigned char>(i,j) = edgeVal;
						if (k >= TEXEDGE_ORIENTATION_REG_NUM/2) conEdge[k- TEXEDGE_ORIENTATION_REG_NUM/2].at<unsigned char>(i,j) = edgeVal;
						else conEdge[k].at<unsigned char>(i,j) = edgeVal;
					//}
				}
			}
		}
	}

	for (int i=0;i<TEXEDGE_ORIENTATION_REG_NUM;i++){
		cv::Mat tmpIm;
		cv::distanceTransform(1-texEdge[i],tmpIm,CV_DIST_L2,5);
		cv::threshold( tmpIm, texEdgeDist[i], threshDist, threshDist,THRESH_TRUNC );
		//cv::Scharr(texEdgeDist[i],texEdgeDistDX[i],CV_32F,1,0);
		//cv::Scharr(texEdgeDist[i],texEdgeDistDY[i],CV_32F,0,1);
	}
	for (int i=0;i<TEXEDGE_ORIENTATION_REG_NUM/2;i++){
		cv::Mat tmpIm;
		cv::distanceTransform(1-conEdge[i],tmpIm,CV_DIST_L2,5);
		cv::threshold( tmpIm, conEdgeDist[i], threshDist, threshDist,THRESH_TRUNC );
		//cv::Scharr(conEdgeDist[i],conEdgeDistDX[i],CV_32F,1,0);
		//cv::Scharr(conEdgeDist[i],conEdgeDistDY[i],CV_32F,0,1);
	}

	//cv::Mat dst,dst2;
	//cv::distanceTransform(1-edgeIm,dst,CV_DIST_L2,5);
	//dst = conEdgeDist[0]*30;
	//dst.convertTo(dst2,CV_8U);
	//cv::applyColorMap(dst2,dst,cv::COLORMAP_JET);
	///// Using Canny's output as a mask, we display our result

	////colorIm.copyTo( dst, edgeIm);
	//imshow( "edgeIm", edgeIm );
	//imshow( "texedges0", texEdge[0] );
	////imshow( "texedgesDist0", dst2 );
	//imshow( "texedges1", texEdge[1] );
	//imshow( "texedges2", texEdge[2] );
	//imshow( "texedges3", texEdge[3] );
	//imshow( "texedges4", texEdge[4] );
	//imshow( "texedges5", texEdge[5] );
	//imshow( "conedges0", conEdge[0] );
	//imshow( "conedges1", conEdge[1] );
	//cv::waitKey();

	return true;
}

float FaceServices2::eE(cv::Mat colorIm, std::vector<cv::Point2f> projPoints, BFMParams &params, int type){
	float err = 0;
	int nPoints = 0;
	for (int i=0;i<projPoints.size();i++){
		int ix = floor(projPoints[i].x+0.5);
		int iy = floor(projPoints[i].y+0.5);
		if (ix >= 0 && iy >= 0 && ix < colorIm.cols && iy < colorIm.rows) {
			if (type == 0) {		// Texture edges
				int bin = params.texEdgeVisBin[i];
                		CvPoint2D64f tmpPt = cvPoint2D64f(projPoints[i].x,projPoints[i].y);
				float val = avSubMatValue32F( &tmpPt, texEdgeDist + bin );
				err += val*val;
				nPoints++;
			}
			else {
				int bin = params.conEdgeBin[i];
                		CvPoint2D64f tmpPt = cvPoint2D64f(projPoints[i].x,projPoints[i].y);
				float val = avSubMatValue32F( &tmpPt, conEdgeDist + bin );
				err += val*val;
				nPoints++;
			}
		}
	}
	return sqrt(err/nPoints);
}

void FaceServices2::initRenderer(cv::Mat &colorIm){
    cv::Mat faces = festimator.getFaces() - 1;
    cv::Mat faces_fill = festimator.getFaces_fill() - 1;
    cv::Mat colors;
    cv::Mat shape = festimator.getShape(cv::Mat::zeros(1,1,CV_32F));
    cv::Mat tex = festimator.getTexture(cv::Mat::zeros(1,1,CV_32F));

    im_render = new FImRenderer(cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3));
    im_render->loadMesh(shape,tex,faces_fill);
}

void FaceServices2::mergeIm(cv::Mat* output,cv::Mat bg,cv::Mat depth){
	for (int i=0;i<bg.rows;i++){
		for (int j=0;j<bg.cols;j++){
			if (depth.at<float>(i,j) >= 0.9999)
				output->at<Vec3b>(i, j) = bg.at<Vec3b>(i,j);
		}
	}
}
float FaceServices2::eS(cv::Mat &alpha, cv::Mat &beta, BFMParams &Params){
    //return 0;
    cv::Vec6d out = vS(alpha, beta);
    //std::cout << "out " << out << std::endl;
    //printf("%f %f %f %f %f %f\n",Params.sS[0],Params.sS[1],Params.sS[2],Params.sS[3],Params.sS[4],Params.sS[5]);
    float sum = 0;
    for (int i=0;i<6;i++)
        sum += Params.sS[i] * Params.sS[i] * out(i) * out(i);
    sum = sqrt(sum);
    //std::cout << "sum " << sum << std::endl;
    return sum;
}


cv::Vec6d FaceServices2::vS(cv::Mat &alpha, cv::Mat &beta){
	cv::Vec6d out(0,0,0,0,0,0);
	cv::Mat tmp = (*(FaceServices2::symSPC))(cv::Rect(0,0,alpha.rows,3)) * alpha;
	for (int i=0;i<3;i++)
		out(i) = tmp.at<float>(i,0);
	tmp = (*(FaceServices2::symTPC))(cv::Rect(0,0,beta.rows,3)) * beta;
	for (int i=0;i<3;i++)
		out(i+3) = tmp.at<float>(i,0);
	return out;
}

//---------------------------------Sym-------------------------------------------------------------------------

float FaceServices2::eS(cv::Mat &alpha, cv::Mat &beta, BFMSymParams &Params){
    //return 0;
    cv::Vec6d out = vS(alpha, beta);
    //std::cout << "out " << out << std::endl;
    //printf("%f %f %f %f %f %f\n",Params.sS[0],Params.sS[1],Params.sS[2],Params.sS[3],Params.sS[4],Params.sS[5]);
    float sum = 0;
    for (int i=0;i<6;i++)
        sum += Params.sS[i] * Params.sS[i] * out(i) * out(i);
    sum = sqrt(sum);
    //std::cout << "sum " << sum << std::endl;
    return sum;
}

void FaceServices2::randSelectTrianglesSym(int numPoints, BFMSymParams &params, std::vector<int> &inds){
	cv::RNG rng(cv::getTickCount());
	for (int i=0;i<numPoints;i++){
		float tmp = rng.uniform(0.0f,1.0f);
		int selected = 0;
		int bin = floor(tmp*NUM_AREA_BIN);
		for (int j=params.indexCSArea[bin];j<params.triAreas[0].rows;j++){
			if (params.trisumAreas.at<float>(j,0) > 0){
				if (params.triCSAreas.at<float>(j,0) > tmp) break;
				selected = j;
			}
		}
		//printf("tmp %f %d %d %d %f\n",tmp,bin,params.indexCSArea[bin], selected, params.triCSAreas.at<float>(selected,0));
		inds.push_back(selected);
	}
}
bool FaceServices2::singleFrameReconSym(cv::Mat colorIm, cv::Mat lms0,cv::Vec6d poseCLM,float conf,cv::Mat lmVis,cv::Mat &shape,cv::Mat &tex,string model_file, string lm_file, string pose_file, string refDir){
	printf("set num of thread\n");
  	omp_set_num_threads(1);
	float renderParams[RENDER_PARAMS_COUNT];
	float renderParams2[RENDER_PARAMS_COUNT];
	Mat k_m(3,3,CV_32F,_k);
	//BaselFaceEstimator festimator;
	BFMSymParams params;
	double time = (double)cv::getTickCount();
	params.init();
	std::vector<cv::Mat> r3Ds, t3Ds;
	std::vector<cv::Mat> r3Ds_2, t3Ds_2;
	std::vector<cv::Mat> ims;
	std::vector<cv::Mat> lmss;
	std::vector<float>  cfds;
	prepareEdgeDistanceMaps(colorIm);
	//time = (double)cv::getTickCount();
	Mat alpha = cv::Mat::zeros(20,1,CV_32F);
	Mat beta = cv::Mat::zeros(20,1,CV_32F);
	Mat exprW = cv::Mat::zeros(29,1,CV_32F);
	Mat alpha_bk, beta_bk, exprW_bk;
	shape = festimator.getShape(alpha);
	tex = festimator.getTexture(beta);
	Mat landModel0 = festimator.getLM(shape,poseCLM(4));
	float bCost, cCost, fCost;
	int bestIter = 0;
	bCost = 10000.0f;
	//write_plyFloat("visLM0.ply",landModel0.t());
	std::vector<int> lmVisInd;
	for (int i=0;i<60;i++){
		if (lmVis.at<int>(i)){
			if (/*(i< 17 || i> 26) &&*/ (i > 16 || abs(poseCLM(4)) <= M_PI/10 || (poseCLM(4) > M_PI/10 && i > 7) || (poseCLM(4) < -M_PI/10 && i < 9)))
				lmVisInd.push_back(i);
		}
	}
	cv::Mat tmpIm = colorIm.clone();
	printf("%f yaw, %f\n",poseCLM(4),M_PI/10);

	if (lmVisInd.size() < 8) return false;
	Mat landModel = cv::Mat( lmVisInd.size(),3,CV_32F);
	Mat landIm = cv::Mat( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landModel.at<float>(i,0) = landModel0.at<float>(ind,0);
		landModel.at<float>(i,1) = landModel0.at<float>(ind,1);
		landModel.at<float>(i,2) = landModel0.at<float>(ind,2);
		landIm.at<float>(i,0) = lms0.at<double>(ind);
		landIm.at<float>(i,1) = lms0.at<double>(ind+landModel0.rows);
		//cv::circle(tmpIm,Point(landIm.at<float>(i,0),landIm.at<float>(i,1)),1,Scalar(255,0,0),1);
	}
	//imwrite("visLM.png",tmpIm);
	//write_plyFloat("visLM.ply",landModel.t());
	//getchar();

	Mat lms = cv::Mat( landModel0.rows,2,CV_64F);
	for (int i=0;i<landModel0.rows;i++){
		lms.at<double>(i,0) = lms0.at<double>(i);
		lms.at<double>(i,1) = lms0.at<double>(i+landModel0.rows);
	}
	cv::Mat r3D, t3D;
	festimator.estimatePose3D(landModel,landIm,k_m,r3D, t3D);
	r3Ds.push_back(r3D.clone());
	t3Ds.push_back(t3D.clone());
	lmss.push_back(lms.clone());
	ims.push_back(colorIm.clone());
	cfds.push_back(conf);

	cv::Mat tmp2,tmp3;
	cv::flip(colorIm,tmp2,1);
	tmp3 = cv::Mat::zeros(tmp2.rows,tmp2.cols,tmp2.type());
	tmp2(cv::Rect(0,0,tmp2.cols-1,tmp2.rows)).copyTo(tmp3(cv::Rect(1,0,tmp2.cols-1,tmp2.rows)));
	ims.push_back(tmp3);


	cv::Mat tmp4 = lms.clone();
	for (int i=0;i<17;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(16-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(16-i,1);
	}
	for (int i=17;i<27;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(26+17-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(26+17-i,1);
	}
	for (int i=27;i<=30;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(i,0);
		tmp4.at<double>(i,1) = lms.at<double>(i,1);
	}
	for (int i=31;i<=35;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(31+35-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(31+35-i,1);
	}
	for (int i=36;i<=39;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(36+45-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(36+45-i,1);
	}
	for (int i=40;i<=41;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(40+47-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(40+47-i,1);
	}
	for (int i=42;i<=45;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(36+45-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(36+45-i,1);
	}
	for (int i=46;i<=47;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(40+47-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(40+47-i,1);
	}
	for (int i=48;i<=54;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(48+54-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(48+54-i,1);
	}
	for (int i=55;i<=59;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(55+59-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(55+59-i,1);
	}
	for (int i=60;i<=64;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(60+64-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(60+64-i,1);
	}
	for (int i=65;i<=67;i++) {
		tmp4.at<double>(i,0) = tmp2.cols - lms.at<double>(65+67-i,0);
		tmp4.at<double>(i,1) = lms.at<double>(65+67-i,1);
	}

	lmss.push_back(tmp4);

	cv::Mat rTmp = r3D.clone();
	rTmp.at<float>(2,0) = -rTmp.at<float>(2,0);
	rTmp.at<float>(1,0) = -rTmp.at<float>(1,0);
	r3Ds.push_back(rTmp);

	cv::Mat tTmp = t3D.clone();
	tTmp.at<float>(0,0) = -tTmp.at<float>(0,0);
	t3Ds.push_back(tTmp);

	cfds.push_back(conf);

	int L = 68;

	std::vector<std::vector<int> > lmVisInds;
	std::vector<cv::Mat> landIms;
	for (int k=0;k<r3Ds.size();k++){
		std::vector<int> lmVisInd;
		int count = 0;
		for (int i=0;i<L;i++){
			//float yaw = r3Ds[k].at<float>(1,0);
			float yaw = -poseCLM(4);
			if (k >= 1) yaw = -yaw;
			if (yaw > M_PI/4) {
				if (i < 9 || (i>16 && i < 22) || (i>26 && i<34) || (i>35 && i< 42) || (i>47 && i< 52) || (i>56 && i< 60))
					lmVisInd.push_back(i);
			}
			else if (yaw < -M_PI/4) {
				if ((i>7 && i < 17) || (i>21 && i < 31) || (i>32 && i<36) || (i>41 && i< 48) || (i>50 && i< 58))
					lmVisInd.push_back(i);
			}
			else {
				if (i < 60 && (i > 16 || abs(yaw) <= M_PI/12 || (yaw > M_PI/12 && i < 9) || (yaw < -M_PI/12 && i > 7)))
					lmVisInd.push_back(i);
			}
		}
		cv::Mat tmpLM(lmVisInd.size(),2,CV_32F);
		for (int i=0;i<lmVisInd.size();i++){
			tmpLM.at<float>(i,0) = lmss[k].at<double>(lmVisInd[i],0);
			tmpLM.at<float>(i,1) = lmss[k].at<double>(lmVisInd[i],1);
		}
		lmVisInds.push_back(lmVisInd);
		landIms.push_back(tmpLM);
	}

	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_R+i] = r3D.at<float>(i,0);
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_T+i] = t3D.at<float>(i,0);
	memcpy(renderParams,params.initR,sizeof(float)*RENDER_PARAMS_COUNT);

	cv::Mat faces = festimator.getFaces() - 1;
	cv::Mat faces_fill = festimator.getFaces_fill() - 1;
	cv::Mat colors;

	im_render = new FImRenderer(cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3));
	im_render->loadMesh(shape,tex,faces_fill);
	memset(params.sF,0,sizeof(float)*NUM_EXTRA_FEATURES);
	
	//FILE* flm = fopen("lmList2a.txt","w");
	//for (int i=0;i<lmVisInds[0].size();i++){
	//	fprintf(flm,"%d %f %f\n",lmVisInds[0][i], landIms[0].at<float>(i,0), landIms[0].at<float>(i,1));
	//}
	//fclose(flm);
	//flm = fopen("lmList2b.txt","w");
	//for (int i=0;i<lmVisInds[1].size();i++){
	//	fprintf(flm,"%d %f %f\n",lmVisInds[1][i], landIms[1].at<float>(i,0), landIms[1].at<float>(i,1));
	//}
	//fclose(flm);
	params.sI = 0.0;
	params.sF[FEATURES_LANDMARK] = 8.0f;
	//params.sF[FEATURES_TEXTURE_EDGE] = 4.0f;
	char text[200];
	Mat alpha0, beta0;
	int iter=0;
	int badCount = 0;
	int M = 99;
	for (int i=0;i<2;i++){
		r3Ds_2.push_back(r3Ds[i].clone());
		t3Ds_2.push_back(t3Ds[i].clone());
	}
	//params.sF[FEATURES_TEXTURE_EDGE] = 6;
	//params.sF[FEATURES_CONTOUR_EDGE] = 6;
	//params.optimizeAB[0] = params.optimizeAB[1] = false;
	memset(params.doOptimize,true,sizeof(bool)*6);
	if (refDir.length() == 0){
		for (;iter<10000;iter++) {
			if (iter%50 == 0) {
				//params.sI += 0.5f;
				//if ( params.sF >= 1.0f)
				//params.sF -= 1.0f;
				//sprintf(text,"tmp_%05d.png",iter);
				//renderFaceSym(text, colorIm,false,  alpha, beta, faces,r3Ds,t3Ds, renderParams,exprW );
				updateTrianglesSym(colorIm,faces,false,  alpha,r3Ds,t3Ds, renderParams, params, exprW );
				cCost = updateHessianMatrixSym(false, alpha,beta,r3Ds,t3Ds,renderParams,faces,ims,lmVisInds,landIms,params, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
				prevEF = cEF;
				//getchar();
			}
			sno_step2Sym(false, alpha, beta,r3Ds,t3Ds, renderParams, faces,ims,lmVisInds,landIms,params,exprW);
			//getchar();
			//params.sF -= 0.002;
		}
		//alpha = alpha_bk.clone();
		//memcpy(renderParams,renderParams2,sizeof(float)*RENDER_PARAMS_COUNT);
		//getchar();
		iter = 10000;
		params.optimizeAB[0] = true;
		params.sF[FEATURES_LANDMARK] = 2;
		//params.sF[FEATURES_TEXTURE_EDGE] = 10;
		params.sF[FEATURES_TEXTURE_EDGE] = 0;
		params.sF[FEATURES_CONTOUR_EDGE] = 15;
		badCount = 0;
		//mstep = 0.01;
		for (;iter<15000;iter++) {
			if (iter%20 == 0) {
				//params.sI += 0.5f;
				//if ( params.sF >= 1.0f)
				//params.sF -= 1.0f;
				//sprintf(text,"tmp_%05d.png",iter);
				//renderFaceSym(text, colorIm,false,  alpha, beta, faces,r3Ds,t3Ds, renderParams,exprW );
				updateTrianglesSym(colorIm,faces,false,  alpha,r3Ds,t3Ds, renderParams, params, exprW );
				cCost = updateHessianMatrixSym(false, alpha,beta,r3Ds,t3Ds,renderParams,faces,ims,lmVisInds,landIms,params, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}

				prevEF = cEF;
				//getchar();
			}
			sno_step2Sym(false, alpha, beta,r3Ds,t3Ds, renderParams, faces,ims,lmVisInds,landIms,params,exprW);
			//getchar();
			//params.sF -= 0.002;
		}
		//alpha = alpha_bk.clone();
		//memcpy(renderParams,renderParams2,sizeof(float)*RENDER_PARAMS_COUNT);

		//getchar();
		mstep = 0.001;
		bCost = 10000;
		iter = 15000;
		mstep = 0.001;
		params.optimizeAB[0] = false;
		params.optimizeAB[1] = false;
		params.optimizeExpr = false;
		params.sF[FEATURES_LANDMARK] = params.sF[FEATURES_TEXTURE_EDGE] = params.sF[FEATURES_CONTOUR_EDGE] = 0;
		params.sI = 15.0;
		badCount = 0;
		params.computeEI = true;
		memset(params.doOptimize,false,sizeof(bool)*6);
		memset(params.doOptimize+6,true,sizeof(bool)*(RENDER_PARAMS_COUNT-6));
		//params.doOptimize[RENDER_PARAMS_AMBIENT+1] = params.doOptimize[RENDER_PARAMS_AMBIENT+2] = false;
		//params.doOptimize[RENDER_PARAMS_DIFFUSE+1] = params.doOptimize[RENDER_PARAMS_DIFFUSE+2] = false;
		for (;iter<17000;iter++) {
			if (iter%50 == 0) {
				//params.sI += 0.5f;
				//if ( params.sF >= 1.0f)
				//params.sF -= 1.0f;
				//sprintf(text,"tmp_%05d.png",iter);
				//renderFaceSym(text, colorIm,false,  alpha, beta, faces,r3Ds,t3Ds, renderParams,exprW );
				updateTrianglesSym(colorIm,faces,false,  alpha,r3Ds,t3Ds, renderParams, params, exprW );
				cCost = updateHessianMatrixSym(false, alpha,beta,r3Ds,t3Ds,renderParams,faces,ims,lmVisInds,landIms,params, exprW);
				if (countFail > 40) {
					countFail = 0;
					break;
				}

				//if (bCost > cEF){
				//	alpha_bk.release();
				//	alpha_bk = alpha.clone();
				//	bCost = cEF;
				//	badCount = 0;
				//	memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
				//}
				//else {
				//	badCount++;
				//}
				//if (badCount > 1) break;
				//std::cout << "hess " << params.hessDiag << std::endl;
				prevEF = cEF;
				//getchar();
			}
			sno_step2Sym(false, alpha, beta,r3Ds,t3Ds, renderParams, faces,ims,lmVisInds,landIms,params,exprW);
			//getchar();
			//params.sF -= 0.002;
		}

		//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
		//std::cout << "Times 1 passed: " << time << std::endl;
		//time = (double)cv::getTickCount();

		iter = 17000;
		alpha0 = alpha.clone();
		beta0 = beta.clone();
		alpha = cv::Mat::zeros(M,1,CV_32F);
		beta = cv::Mat::zeros(M,1,CV_32F);
		for (int i=0;i<alpha0.rows; i++) alpha.at<float>(i,0) = alpha0.at<float>(i,0);
		for (int i=0;i<beta0.rows; i++) beta.at<float>(i,0) = beta0.at<float>(i,0);
		alpha_bk = alpha.clone();
		beta_bk = beta.clone();
		exprW_bk = exprW.clone();
		badCount = 0;

		//for (;iter<3000;iter++) {
		//	if (iter%500 == 0) {
		//		sprintf(text,"tmp_%05d.png",iter);
		//		renderFace(text, colorIm, alpha, beta, faces, renderParams );
		//		updateTriangles(colorIm,faces, alpha, renderParams, params );
		//		updateHessianMatrix(alpha,beta,renderParams,faces,colorIm,lmVisInd,landIm,params);
		//		std::cout << "hess " << params.hessDiag << std::endl;
		//		getchar();
		//	}
		//	sno_step(alpha, beta, renderParams, faces,colorIm,lmVisInd,landIm,params);
		//	//getchar();
		//	//params.sF -= 0.002;
		//	//params.sI += 0.002;
		//}
		params.sI = 15.0f;
		params.sF[FEATURES_TEXTURE_EDGE] = 0;
		//params.sF[FEATURES_TEXTURE_EDGE] = 0;
		//params.sF[FEATURES_LANDMARK] = 5.0;
		//params.sF[FEATURES_TEXTURE_EDGE] = 3.5;
		params.sF[FEATURES_CONTOUR_EDGE] = 5.0f;
		params.optimizeAB[0] = params.optimizeAB[1] = true;
		params.optimizeExpr = true;
		mlambda = 0.005;
		memset(params.doOptimize,true,sizeof(bool)*RENDER_PARAMS_COUNT);
		//params.doOptimize[RENDER_PARAMS_AMBIENT+1] = params.doOptimize[RENDER_PARAMS_AMBIENT+2] = false;
		//params.doOptimize[RENDER_PARAMS_DIFFUSE+1] = params.doOptimize[RENDER_PARAMS_DIFFUSE+2] = false;
		//params.doOptimize[RENDER_PARAMS_GAIN] = params.doOptimize[RENDER_PARAMS_GAIN+1] = params.doOptimize[RENDER_PARAMS_GAIN+2] = false;
		for (;iter<20000;iter++) {
			//if (iter % 4000 == 0){
			//	M += 10;
			//	alpha0 = alpha.clone();
			//	beta0 = beta.clone();
			//	alpha = cv::Mat::zeros(M,1,CV_32F);
			//	beta = cv::Mat::zeros(M,1,CV_32F);
			//	for (int i=0;i<alpha0.rows; i++) alpha.at<float>(i,0) = alpha0.at<float>(i,0);
			//	for (int i=0;i<beta0.rows; i++) beta.at<float>(i,0) = beta0.at<float>(i,0);
			//}

			if (iter%250 == 0) {
				//if (iter>4000) memset(params.doOptimize,false,sizeof(bool)*RENDER_PARAMS_COUNT);
				//if (iter>=18500 && conf < 0.95) {
				//	params.sF[FEATURES_LANDMARK] = (conf - 0.85)/0.1 * 5.0f;
				////	//printf("conf %f %f\n",conf,params.sF);
				//	if (params.sF[FEATURES_LANDMARK] < 0) 
				//		params.sF[FEATURES_LANDMARK] = 0;
				//}
				//if (iter>=15000) params.sF = 0;
				//if (iter>=3000) params.sF = 0;
				//if ( params.sF >= 1.0f)
				//params.sI += 1.0f;
				//if ( params.sF >= 5.0f)
				//params.sF -= 0.5f;
				//sprintf(text,"tmp_%05d.png",iter);
				//renderFaceSym(text, colorIm,false,  alpha, beta, faces,r3Ds,t3Ds, renderParams,exprW );
				updateTrianglesSym(colorIm,faces,false,  alpha,r3Ds,t3Ds, renderParams, params, exprW );
				cCost = updateHessianMatrixSym(false, alpha,beta,r3Ds,t3Ds,renderParams,faces,ims,lmVisInds,landIms,params, exprW);
				if (iter == 17000) fCost = cCost;
				if (bCost > cCost || bestIter <= 17000){
					alpha_bk.release(); beta_bk.release(); exprW_bk.release();
					alpha_bk = alpha.clone();
					beta_bk = beta.clone();
					exprW_bk = exprW.clone();
					bCost = cCost;
					badCount = 0;
					bestIter = iter;
					memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
				}
				else {
					badCount++;
				}
			}
			sno_step2Sym(false, alpha, beta,r3Ds,t3Ds, renderParams, faces,ims,lmVisInds,landIms,params,exprW);
		}
/*
		mlambda /= 5;
		for (;iter<23000;iter++) {
			//if (iter % 4000 == 0){
			//	M += 10;
			//	alpha0 = alpha.clone();
			//	beta0 = beta.clone();
			//	alpha = cv::Mat::zeros(M,1,CV_32F);
			//	beta = cv::Mat::zeros(M,1,CV_32F);
			//	for (int i=0;i<alpha0.rows; i++) alpha.at<float>(i,0) = alpha0.at<float>(i,0);
			//	for (int i=0;i<beta0.rows; i++) beta.at<float>(i,0) = beta0.at<float>(i,0);
			//}

			if (iter%250 == 0) {
				//if (iter>4000) memset(params.doOptimize,false,sizeof(bool)*RENDER_PARAMS_COUNT);
				//if (iter>=18500 && conf < 0.95) {
				//	params.sF[FEATURES_LANDMARK] = (conf - 0.85)/0.1 * 5.0f;
				////	//printf("conf %f %f\n",conf,params.sF);
				//	if (params.sF[FEATURES_LANDMARK] < 0) 
				//		params.sF[FEATURES_LANDMARK] = 0;
				//}
				//if (iter>=15000) params.sF = 0;
				//if (iter>=3000) params.sF = 0;
				//if ( params.sF >= 1.0f)
				//params.sI += 1.0f;
				//if ( params.sF >= 5.0f)
				//params.sF -= 0.5f;
				sprintf(text,"tmp_%05d.png",iter);
				renderFaceSym(text, colorIm,false,  alpha, beta, faces,r3Ds,t3Ds, renderParams,exprW );
				updateTrianglesSym(colorIm,faces,false,  alpha,r3Ds,t3Ds, renderParams, params, exprW );
				cCost = updateHessianMatrixSym(false, alpha,beta,r3Ds,t3Ds,renderParams,faces,ims,lmVisInds,landIms,params, exprW);
				if (iter == 17000) fCost = cCost;
				if (bCost > cCost || bestIter <= 17000){
					alpha_bk.release(); beta_bk.release(); exprW_bk.release();
					alpha_bk = alpha.clone();
					beta_bk = beta.clone();
					exprW_bk = exprW.clone();
					bCost = cCost;
					badCount = 0;
					bestIter = iter;
					memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
				}
				else {
					badCount++;
				}
			}
			sno_step2Sym(false, alpha, beta,r3Ds,t3Ds, renderParams, faces,ims,lmVisInds,landIms,params,exprW);
		}
		
		mlambda /= 5;
		for (;iter<26000;iter++) {
			//if (iter % 4000 == 0){
			//	M += 10;
			//	alpha0 = alpha.clone();
			//	beta0 = beta.clone();
			//	alpha = cv::Mat::zeros(M,1,CV_32F);
			//	beta = cv::Mat::zeros(M,1,CV_32F);
			//	for (int i=0;i<alpha0.rows; i++) alpha.at<float>(i,0) = alpha0.at<float>(i,0);
			//	for (int i=0;i<beta0.rows; i++) beta.at<float>(i,0) = beta0.at<float>(i,0);
			//}

			if (iter%250 == 0) {
				//if (iter>4000) memset(params.doOptimize,false,sizeof(bool)*RENDER_PARAMS_COUNT);
				//if (iter>=18500 && conf < 0.95) {
				//	params.sF[FEATURES_LANDMARK] = (conf - 0.85)/0.1 * 5.0f;
				////	//printf("conf %f %f\n",conf,params.sF);
				//	if (params.sF[FEATURES_LANDMARK] < 0) 
				//		params.sF[FEATURES_LANDMARK] = 0;
				//}
				//if (iter>=15000) params.sF = 0;
				//if (iter>=3000) params.sF = 0;
				//if ( params.sF >= 1.0f)
				//params.sI += 1.0f;
				//if ( params.sF >= 5.0f)
				//params.sF -= 0.5f;
				sprintf(text,"tmp_%05d.png",iter);
				renderFaceSym(text, colorIm,false,  alpha, beta, faces,r3Ds,t3Ds, renderParams,exprW );
				updateTrianglesSym(colorIm,faces,false,  alpha,r3Ds,t3Ds, renderParams, params, exprW );
				cCost = updateHessianMatrixSym(false, alpha,beta,r3Ds,t3Ds,renderParams,faces,ims,lmVisInds,landIms,params, exprW);
				if (iter == 17000) fCost = cCost;
				if (bCost > cCost || bestIter <= 17000){
					alpha_bk.release(); beta_bk.release(); exprW_bk.release();
					alpha_bk = alpha.clone();
					beta_bk = beta.clone();
					exprW_bk = exprW.clone();
					bCost = cCost;
					badCount = 0;
					bestIter = iter;
					memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
				}
				else {
					badCount++;
				}
			}
			sno_step2Sym(false, alpha, beta,r3Ds,t3Ds, renderParams, faces,ims,lmVisInds,landIms,params,exprW);
		}
*/
		//sprintf(text,"tmp_%05d.png",iter);
		//renderFaceSym(text, colorIm,false,  alpha, beta, faces,r3Ds,t3Ds, renderParams,exprW );
		updateTrianglesSym(colorIm,faces,false,  alpha,r3Ds,t3Ds, renderParams, params, exprW );
		cCost = updateHessianMatrixSym(false, alpha,beta,r3Ds,t3Ds,renderParams,faces,ims,lmVisInds,landIms,params, exprW);
		if (bCost > cCost){
			alpha_bk.release(); beta_bk.release(); exprW_bk.release();
			alpha_bk = alpha.clone();
			beta_bk = beta.clone();
			exprW_bk = exprW.clone();
			bCost = cCost;
			badCount = 0;
			bestIter = iter;
			memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
		}

		for (int i=0;i<3;i++)
			renderParams2[RENDER_PARAMS_R+i] = r3Ds[0].at<float>(i,0);
		for (int i=0;i<3;i++)
			renderParams2[RENDER_PARAMS_T+i] = t3Ds[0].at<float>(i,0);
	}
	else {
		int EM = 29;
		loadReference(refDir, model_file, alpha, beta, renderParams, M, exprW, EM);

		//std::cout << "alpha " << alpha << std::endl;
		//std::cout << "beta " << beta << std::endl;
		//getchar();
		params.sI = 15.0f;
		params.sF[FEATURES_LANDMARK] = 8.0f;
		//if (conf < 0.95) {
		//	params.sF[FEATURES_LANDMARK] = (conf - 0.85)/0.1 * 8.0f;
		//	//printf("conf %f %f\n",conf,params.sF);
		//	if (params.sF[FEATURES_LANDMARK] < 0) 
		//		params.sF[FEATURES_LANDMARK] = 0;
		//}
		params.computeEI = true;
		params.optimizeAB[0] = params.optimizeAB[1] = true;
		memset(params.doOptimize,true,sizeof(bool)*RENDER_PARAMS_COUNT);
		//params.doOptimize[RENDER_PARAMS_AMBIENT+1] = params.doOptimize[RENDER_PARAMS_AMBIENT+2] = false;
		//params.doOptimize[RENDER_PARAMS_DIFFUSE+1] = params.doOptimize[RENDER_PARAMS_DIFFUSE+2] = false;
		//params.doOptimize[RENDER_PARAMS_GAIN] = params.doOptimize[RENDER_PARAMS_GAIN+1] = params.doOptimize[RENDER_PARAMS_GAIN+2] = false;
		//time = (double)cv::getTickCount();
		for (iter = 0;iter<1000;iter++) {
			if (iter%250 == 0) {
				updateTrianglesSym(colorIm,faces,false,  alpha,r3Ds,t3Ds, renderParams, params, exprW );
				cCost = updateHessianMatrixSym(false, alpha,beta,r3Ds,t3Ds,renderParams,faces,ims,lmVisInds,landIms,params, exprW);
				if (iter == 0) fCost = cCost;
				if (bCost > cCost || iter == 0){
					alpha_bk.release(); beta_bk.release(); 
					alpha_bk = alpha.clone();
					beta_bk = beta.clone();
					exprW_bk.release();
					exprW_bk = exprW.clone();
					bCost = cCost;
					badCount = 0;
					bestIter = iter;
					memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
				}
				else {
					badCount++;
				}
			}
			sno_step2Sym(false, alpha, beta,r3Ds,t3Ds, renderParams, faces,ims,lmVisInds,landIms,params,exprW);
		}

		sprintf(text,"tmp_%05d.png",iter);
		renderFaceSym(text, colorIm,false,  alpha, beta, faces,r3Ds,t3Ds, renderParams,exprW );
		updateTrianglesSym(colorIm,faces,false,  alpha,r3Ds,t3Ds, renderParams, params, exprW );
		cCost = updateHessianMatrixSym(false, alpha,beta,r3Ds,t3Ds,renderParams,faces,ims,lmVisInds,landIms,params, exprW);
		if (bCost > cCost){
			alpha_bk.release(); beta_bk.release(); 
			alpha_bk = alpha.clone();
			beta_bk = beta.clone();
			exprW_bk.release();
			exprW_bk = exprW.clone();
			bCost = cCost;
			badCount = 0;
			bestIter = iter;
			memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
		}

		for (int i=0;i<3;i++)
			renderParams2[RENDER_PARAMS_R+i] = r3Ds[0].at<float>(i,0);
		for (int i=0;i<3;i++)
			renderParams2[RENDER_PARAMS_T+i] = t3Ds[0].at<float>(i,0);
	}
	time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
	std::cout << "Times 2 passed: " << time << std::endl;
	shape = festimator.getShape(alpha, exprW);
	tex = festimator.getTexture(beta);
	sprintf(text,"%s",model_file.c_str());
	write_plyFloat(text,shape,tex,faces);

	sprintf(text,"%s.alpha",model_file.c_str());
	FILE* ff=fopen(text,"w");
	for (int i=0;i<alpha.rows;i++) fprintf(ff,"%f\n",alpha.at<float>(i,0));
	fclose(ff);
	sprintf(text,"%s.beta",model_file.c_str());
	ff=fopen(text,"w");
	for (int i=0;i<beta.rows;i++) fprintf(ff,"%f\n",beta.at<float>(i,0));
	fclose(ff);
	sprintf(text,"%s.rend",model_file.c_str());
	ff=fopen(text,"w");
	for (int i=0;i<RENDER_PARAMS_COUNT;i++) fprintf(ff,"%f\n",renderParams[i]);
	fclose(ff);
	sprintf(text,"%s.expr",model_file.c_str());
	ff=fopen(text,"w");
	for (int i=0;i<exprW.rows;i++) fprintf(ff,"%f\n",exprW.at<float>(i,0));
	fclose(ff);
	//float err = eS(alpha, beta, params);
	//sprintf(text,"%s.sym",model_file.c_str());
	//ff=fopen(text,"w");
	//fprintf(ff,"%f\n",err);
	//fclose(ff);

	sprintf(text,"%s",pose_file.c_str());
	ff=fopen(text,"w");
	fprintf(ff,"%f, %f, %f, %f, %f, %f\n",renderParams[0],renderParams[1],renderParams[2], renderParams[3], renderParams[4], renderParams[5]);
	fclose(ff);

	shape.release(); tex.release();	
	return true;
}


cv::Mat FaceServices2::computeGradientSym(bool part, cv::Mat &alpha, cv::Mat &beta,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat faces,std::vector<cv::Mat> ims,std::vector<std::vector<int> > lmInds, std::vector<cv::Mat> landIms, BFMSymParams &params, std::vector<int> &inds, cv::Mat exprW){
	int M = alpha.rows;
	int EM = exprW.rows;
	int nTri = 60;
	float step;
	double time;
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	cv::Mat out(2*M+EM+RENDER_PARAMS_COUNT,1,CV_32F);

	cv::Mat alpha2, beta2, expr2;
	cv::Mat centerPoints[2], centerTexs, normals[2];
	cv::Mat ccenterPoints[2], ccenterTexs, cnormals[2];
	cv::Mat texEdge3D[2],conEdge3D[2],texEdge3D2[2], conEdge3D2[2];
	std::vector<cv::Point2f> pPoints[2];
	float renderParams2[RENDER_PARAMS_COUNT];
	//std::vector<cv::Mat> r3Ds_2, t3Ds_2;

	//cv::Mat rVec(3,1,CV_32F,renderParams+RENDER_PARAMS_R);
	//cv::Mat tVec(3,1,CV_32F,renderParams+RENDER_PARAMS_T);
	//cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	//cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	cES = eS(alpha, beta, params);
	//printf("%f\n",cES);
	float currEF = eFSym(part, alpha, lmInds, landIms, r3Ds, t3Ds, renderParams,exprW);
	cETE = cECE = 0;
	if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
		if (part) {
			texEdge3D[0] = festimator.getTriByAlphaParts(alpha,params.texEdgeVisIndices[0],exprW);
			texEdge3D[1] = festimator.getTriByAlphaPartsFlipExpr(alpha,params.texEdgeVisIndices[1],exprW);
		}
		else {
			texEdge3D[0] = festimator.getTriByAlpha(alpha,params.texEdgeVisIndices[0],exprW);
			texEdge3D[1] = festimator.getTriByAlphaFlipExpr(alpha,params.texEdgeVisIndices[1],exprW);
		}
		projectPoints(texEdge3D[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
		projectPoints(texEdge3D[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
		cETE = eESym(ims[0],pPoints,params,0);
	}
	if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
		if (part) {
			conEdge3D[0] = festimator.getTriByAlphaParts(alpha,params.conEdgeIndices[0],exprW);
			conEdge3D[1] = festimator.getTriByAlphaPartsFlipExpr(alpha,params.conEdgeIndices[1],exprW);
		}
		else {
			conEdge3D[0] = festimator.getTriByAlpha(alpha,params.conEdgeIndices[0],exprW);
			conEdge3D[1] = festimator.getTriByAlphaFlipExpr(alpha,params.conEdgeIndices[1],exprW);
		}
		projectPoints(conEdge3D[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
		projectPoints(conEdge3D[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
		cECE = eESym(ims[0],pPoints,params,1);
	}
	randSelectTrianglesSym(nTri, params, inds);
	if (params.computeEI) {
		getTrianglesCenterNormalSym(part, alpha, beta,  faces, inds, centerPoints, centerTexs, normals,exprW);

		ccenterPoints[0] = centerPoints[0].clone();
		ccenterPoints[1] = centerPoints[1].clone();
		ccenterTexs = centerTexs.clone();
		cnormals[0] = normals[0].clone();
		cnormals[1] = normals[1].clone();
	}
	float currEI = eISym(ims, centerPoints, centerTexs, normals, r3Ds, t3Ds,renderParams,inds,params,exprW);
	cEI = currEI;
	//if (params.computeEI) getchar();
	//printf("currEF %f %f\n",currEF,currEI);
	cEF = currEF;
	//if (cEF > prevEF) printf("grad -> %f vs. ",currEF);
	// alpha
	step = mstep*20;
	if (params.optimizeAB[0]) {
//#pragma omp parallel for
		for (int i=0;i<M; i++){
			cv::Mat centerPoints[2], normals[2], texEdge3D2[2], conEdge3D2[2];
			std::vector<cv::Point2f> pPoints[2];
			cv::Mat alpha2 = alpha.clone();
			alpha2.at<float>(i,0) += step;
			float tmpEF = eFSym(part, alpha2, lmInds, landIms, r3Ds, t3Ds, renderParams,exprW);
			//if (i==M-1) 	time = (double)cv::getTickCount();
			if (params.computeEI) getTrianglesCenterVNormalSym(part, alpha2,  faces, inds, centerPoints, normals,exprW);
			float tmpEI = eISym(ims, centerPoints, ccenterTexs, normals, r3Ds, t3Ds,renderParams,inds,params,exprW);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) {
					texEdge3D2[0] = festimator.getTriByAlphaParts(alpha2,params.texEdgeVisIndices[0],exprW);
					texEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha2,params.texEdgeVisIndices[1],exprW);
				}
				else {
					texEdge3D2[0] = festimator.getTriByAlpha(alpha2,params.texEdgeVisIndices[0],exprW);
					texEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha2,params.texEdgeVisIndices[1],exprW);
				}
				projectPoints(texEdge3D2[0],r3Ds[0], t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D2[1],r3Ds[1], t3Ds[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE = cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) {
					conEdge3D2[0] = festimator.getTriByAlphaParts(alpha2,params.conEdgeIndices[0],exprW);
					conEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha2,params.conEdgeIndices[1],exprW);
				}
				else {
					conEdge3D2[0] = festimator.getTriByAlpha(alpha2,params.conEdgeIndices[0],exprW);
					conEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha2,params.conEdgeIndices[1],exprW);
				}
				projectPoints(conEdge3D2[0],r3Ds[0], t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D2[1],r3Ds[1], t3Ds[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE = cECE2 - cECE;
			}
			float cES2 = eS(alpha2, beta, params);
			out.at<float>(i,0) = (params.sI * (tmpEI - currEI) + params.sF[FEATURES_LANDMARK] * (tmpEF - currEF))/step
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/step 
				+ 2*alpha.at<float>(i,0)/(0.25f*M) + cES2 - cES;
			//if (cEF > prevEF) printf("%f, ",tmpEF);
		}
	}
	// beta
	step = mstep*20;
	if (params.optimizeAB[1]) {
//#pragma omp parallel for
		for (int i=0;i<M; i++){
			cv::Mat centerTexs;
			cv::Mat beta2 = beta.clone();
			beta2.at<float>(i,0) += step;
			if (params.computeEI)
				getTrianglesCenterTex(part, beta2,  faces, inds, centerTexs);
			float tmpEI = eISym(ims, ccenterPoints, centerTexs, cnormals, r3Ds, t3Ds,renderParams,inds,params, exprW);
			float cES2 = eS(alpha, beta2, params);
			out.at<float>(M+i,0) = (params.sI * (tmpEI - currEI))/step + 2*beta.at<float>(i,0)/(0.5f*M) + cES2 - cES;
		}
	}
	// expr
	step = mstep*5;
	if (params.optimizeExpr) {
//#pragma omp parallel for
		for (int i=0;i<EM; i++){
			cv::Mat centerPoints[2], normals[2], texEdge3D2[2], conEdge3D2[2];
			std::vector<cv::Point2f> pPoints[2];
			cv::Mat expr2 = exprW.clone();
			expr2.at<float>(i,0) += step;
			float tmpEF = eFSym(part, alpha, lmInds, landIms, r3Ds, t3Ds, renderParams,expr2);
			//if (i==M-1) 	time = (double)cv::getTickCount();
			if (params.computeEI) getTrianglesCenterVNormalSym(part, alpha,  faces, inds, centerPoints, normals,expr2);
			float tmpEI = eISym(ims, centerPoints, ccenterTexs, normals, r3Ds, t3Ds,renderParams,inds,params,expr2);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) {
					texEdge3D2[0] = festimator.getTriByAlphaParts(alpha,params.texEdgeVisIndices[0],expr2);
					texEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha,params.texEdgeVisIndices[1],expr2);
				}
				else {
					texEdge3D2[0] = festimator.getTriByAlpha(alpha,params.texEdgeVisIndices[0],expr2);
					texEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha,params.texEdgeVisIndices[1],expr2);
				}
				projectPoints(texEdge3D2[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D2[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE = cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) {
					conEdge3D2[0] = festimator.getTriByAlphaParts(alpha,params.conEdgeIndices[0],expr2);
					conEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha,params.conEdgeIndices[1],expr2);
				}
				else {
					conEdge3D2[0] = festimator.getTriByAlpha(alpha,params.conEdgeIndices[0],expr2);
					conEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha,params.conEdgeIndices[1],expr2);
				}
				projectPoints(conEdge3D2[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D2[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE = cECE2 - cECE;
			}
			out.at<float>(2*M+i,0) = (params.sI * (tmpEI - currEI) + params.sF[FEATURES_LANDMARK] * (tmpEF - currEF))/step
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/step 
				+ params.sExpr * 2*exprW.at<float>(i,0)/(0.25f*29);
			//if (M == 99) out.at<float>(2*M+i,0) /= 100;
			//if (cEF > prevEF) printf("%f, ",tmpEF);
		}
	}
	// r
	//step = 0.05;
	//step = 0.01;
	step = mstep*2;
	//step = 0.01;
	if (params.doOptimize[RENDER_PARAMS_R]) {
		std::vector<cv::Mat> r3Ds_2;
		for (int k=0;k<2; k++){
			r3Ds_2.push_back(r3Ds[k].clone());
		}
		for (int i=0;i<3; i++){
			r3Ds_2[0].at<float>(i,0) += step;
			if (i == 1 || i==2) r3Ds_2[1].at<float>(i,0) = -r3Ds_2[0].at<float>(i,0);
			else r3Ds_2[1].at<float>(i,0) = r3Ds_2[0].at<float>(i,0);
			//memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			//renderParams2[RENDER_PARAMS_R+i] += step;
			float tmpEF = eFSym(part, alpha, lmInds, landIms, r3Ds_2, t3Ds, renderParams,exprW);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D[0],r3Ds_2[0], t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D[1],r3Ds_2[1], t3Ds[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE = cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D[0],r3Ds_2[0], t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D[1],r3Ds_2[1], t3Ds[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE = cECE2 - cECE;
			}
			float tmpEI = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds_2, t3Ds, renderParams,inds,params, exprW);
			out.at<float>(2*M+EM+i,0) = (params.sI * (tmpEI - currEI) + params.sF[FEATURES_LANDMARK] * (tmpEF - currEF))/step 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/step 
				+ 2*(r3Ds[0].at<float>(i,0) - params.initR[RENDER_PARAMS_R+i])/params.sR[RENDER_PARAMS_R+i];
			r3Ds_2[0].at<float>(i,0) = r3Ds[0].at<float>(i,0);
			r3Ds_2[1].at<float>(i,0) = r3Ds[1].at<float>(i,0);
			//if (cEF > prevEF) printf("%f, ",tmpEF);
		}
	}
	// t
	step = mstep*10;
	//step = 1;
	//step = 0.1;
	if (params.doOptimize[RENDER_PARAMS_T]) {
		std::vector<cv::Mat> t3Ds_2;
		for (int k=0;k<2; k++){
			t3Ds_2.push_back(t3Ds[k].clone());
		}
		for (int i=0;i<3; i++){
			t3Ds_2[0].at<float>(i,0) += step;
			if (i == 0) t3Ds_2[1].at<float>(i,0) = -t3Ds_2[0].at<float>(i,0);
			else t3Ds_2[1].at<float>(i,0) = t3Ds_2[0].at<float>(i,0);
			//memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			//renderParams2[RENDER_PARAMS_T+i] += step;
			float tmpEF = eFSym(part, alpha, lmInds, landIms, r3Ds, t3Ds_2, renderParams,exprW);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D[0],r3Ds[0], t3Ds_2[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D[1],r3Ds[1], t3Ds_2[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE = cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D[0],r3Ds[0], t3Ds_2[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D[1],r3Ds[1], t3Ds_2[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE = cECE2 - cECE;
			}
			float tmpEI = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds_2, renderParams,inds,params, exprW);
			out.at<float>(2*M+EM+RENDER_PARAMS_T+i,0) = (params.sI * (tmpEI - currEI) + params.sF[FEATURES_LANDMARK] * (tmpEF - currEF))/step 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/step 
				+ 2*(t3Ds[0].at<float>(i,0) - params.initR[RENDER_PARAMS_T+i])/params.sR[RENDER_PARAMS_T+i];
			//if (cEF > prevEF) printf("%f, ",tmpEF);
			t3Ds_2[0].at<float>(i,0) = t3Ds[0].at<float>(i,0);
			t3Ds_2[1].at<float>(i,0) = t3Ds[1].at<float>(i,0);
		}
	}
	// AMBIENT
	step = mstep;
	if (params.doOptimize[RENDER_PARAMS_AMBIENT]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_AMBIENT+i] += step;
			float tmpEI = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds, renderParams2,inds,params, exprW);
			out.at<float>(2*M+EM+RENDER_PARAMS_AMBIENT+i,0) = (params.sI * (tmpEI - currEI))/step + 2*(renderParams[RENDER_PARAMS_AMBIENT+i] - params.initR[RENDER_PARAMS_AMBIENT+i])/params.sR[RENDER_PARAMS_AMBIENT+i];
			//out.at<float>(2*M+RENDER_PARAMS_AMBIENT+i,0) = 0;
		}
	}
	// DIFFUSE
	//step = 0.02;
	if (params.doOptimize[RENDER_PARAMS_DIFFUSE]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_DIFFUSE+i] += step;
			float tmpEI = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds,renderParams2,inds,params, exprW);
			out.at<float>(2*M+EM+RENDER_PARAMS_DIFFUSE+i,0) = (params.sI * (tmpEI - currEI))/step + 2*(renderParams[RENDER_PARAMS_DIFFUSE+i] - params.initR[RENDER_PARAMS_DIFFUSE+i])/params.sR[RENDER_PARAMS_DIFFUSE+i];
			//out.at<float>(2*M+RENDER_PARAMS_DIFFUSE+i,0) = 0;
		}
	}
	// LDIR
	//step = 0.02;
	step = 0.01;
	if (params.doOptimize[RENDER_PARAMS_LDIR]) {
		for (int i=0;i<2; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_LDIR+i] += step;
			float tmpEI = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds,renderParams2,inds,params, exprW);
			out.at<float>(2*M+EM+RENDER_PARAMS_LDIR+i,0) = (params.sI * (tmpEI - currEI))/step + 2*(renderParams[RENDER_PARAMS_LDIR+i] - params.initR[RENDER_PARAMS_LDIR+i])/params.sR[RENDER_PARAMS_LDIR+i];
			//out.at<float>(2*M+RENDER_PARAMS_LDIR+i,0) = 0;
		}
	}
	// others
	//step = 0.01;
	step = mstep;
	for (int i=RENDER_PARAMS_CONTRAST;i<RENDER_PARAMS_COUNT; i++){
		//out.at<float>(2*M+i,0) = 0;
		if (params.doOptimize[i]) {
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[i] += step;
			float tmpEI = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds,renderParams2,inds,params, exprW);
			out.at<float>(2*M+EM+i,0) = (params.sI * (tmpEI - currEI))/step + 2*(renderParams[i] - params.initR[i])/params.sR[i];
		}
	}
	
	ccenterPoints[0].release(); ccenterTexs.release(); cnormals[0].release();
	ccenterPoints[1].release(); cnormals[1].release();
	return out;
}

void FaceServices2::sno_step2Sym(bool part, cv::Mat &alpha, cv::Mat &beta,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat faces,std::vector<cv::Mat> ims,std::vector<std::vector<int> > lmInds, std::vector<cv::Mat> landIms, BFMSymParams &params, cv::Mat &exprW){
	float lambda = /*0.005;*/ mlambda;
	std::vector<int> inds;
	cv::Mat dE = computeGradientSym(part, alpha, beta, r3Ds, t3Ds, renderParams, faces, ims, lmInds, landIms, params,inds, exprW);
	cv::Mat dirMove = dE*0;

	int M = alpha.rows;
	int EM = exprW.rows;
	if (params.optimizeAB[0]){
		for (int i=0;i<M;i++)
			if (abs(params.hessDiag.at<float>(i,0)) > 0.0000001) {
				dirMove.at<float>(i,0) = - lambda*dE.at<float>(i,0)/abs(params.hessDiag.at<float>(i,0));
			}
	}
	if (params.optimizeAB[1]){
		for (int i=0;i<M;i++)
			if (abs(params.hessDiag.at<float>(M+i,0)) > 0.0000001) {
				dirMove.at<float>(M+i,0) = - lambda*dE.at<float>(M+i,0)/abs(params.hessDiag.at<float>(M+i,0));
			}
	}
	if (params.optimizeExpr){
		for (int i=0;i<EM;i++)
			if (abs(params.hessDiag.at<float>(2*M+i,0)) > 0.0000001) {
				dirMove.at<float>(2*M+i,0) = - lambda*dE.at<float>(2*M+i,0)/abs(params.hessDiag.at<float>(2*M+i,0));
			}
	}

	for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
		if (params.doOptimize[i]){
			if (abs(params.hessDiag.at<float>(2*M+EM+i,0)) > 0.0000001) {
				dirMove.at<float>(2*M+EM+i,0) = - lambda*dE.at<float>(2*M+EM+i,0)/abs(params.hessDiag.at<float>(2*M+EM+i,0));
			}
		}
	}

	float pc = line_searchSym(part, alpha, beta, r3Ds, t3Ds, renderParams, dirMove,inds, faces, ims, lmInds, landIms, params, exprW, 20);
	
	////if (params.optimizeAB[0] && params.optimizeAB[1]){
	//	printf("pc = %f\n",pc);
	//	printf("Shape: ");
	//	for (int i=0;i<10;i++) printf("%f ", dirMove.at<float>(i,0));
	//	printf("\n");
	////	printf("Texture: ");
	////	for (int i=0;i<10;i++) printf("%f ", dirMove.at<float>(M+i,0));
	////	printf("\n");
	//	printf("Expr: ");
	//	for (int i=0;i<29;i++) printf("%f ", dirMove.at<float>(2*M+i,0));
	//	printf("\n");
	//	printf("R t: ");
	//	for (int i=0;i<6;i++) printf("%f ", dirMove.at<float>(2*M+EM+i,0));
	//	getchar();
	////}
	if (pc == 0) countFail++;
	else {
		if (params.optimizeAB[0]){
			for (int i=0;i<M;i++) {
				alpha.at<float>(i,0) += pc*dirMove.at<float>(i,0);
				if (alpha.at<float>(i,0) > maxVal) alpha.at<float>(i,0) = maxVal;
				else if (alpha.at<float>(i,0) < -maxVal) alpha.at<float>(i,0) = -maxVal;
			}
		}
		if (params.optimizeAB[1]){
			for (int i=0;i<M;i++) {
				beta.at<float>(i,0) += pc*dirMove.at<float>(M+i,0);
				if (beta.at<float>(i,0) > maxVal) beta.at<float>(i,0) = maxVal;
				else if (beta.at<float>(i,0) < -maxVal) beta.at<float>(i,0) = -maxVal;
			}
		}
		if (params.optimizeExpr){
			for (int i=0;i<EM;i++) {
				exprW.at<float>(i,0) += pc*dirMove.at<float>(i+2*M,0);
				if (exprW.at<float>(i,0) > 3) exprW.at<float>(i,0) = 3;
				else if (exprW.at<float>(i,0) < -3) exprW.at<float>(i,0) = -3;
			}
		}

		if (params.doOptimize[RENDER_PARAMS_R]){
			for (int i=0;i<3;i++) {
				renderParams[RENDER_PARAMS_R + i] = r3Ds[0].at<float>(i,0) + pc*dirMove.at<float>(2*M+EM+EM+RENDER_PARAMS_T+i,0);
				t3Ds[0].at<float>(i,0) = renderParams[RENDER_PARAMS_T + i];
				if (i == 0) t3Ds[1].at<float>(i,0) = -renderParams[RENDER_PARAMS_T + i];
				else t3Ds[1].at<float>(i,0) = renderParams[RENDER_PARAMS_T + i];
			}
		}
		for (int i=6;i<RENDER_PARAMS_COUNT;i++) {
			if (params.doOptimize[i]){
				renderParams[i] += pc*dirMove.at<float>(2*M+EM+i,0);
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (renderParams[i] > 1.0) renderParams[i] = 1.0;
					if (renderParams[i] < 0) renderParams[i] = 0;

				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (renderParams[i] > 3.0) renderParams[i]  = 3;
					if (renderParams[i] < 0.3) renderParams[i] = 0.3;
				}
			}
		}
	}
	prevEF = cEF;
}

float FaceServices2::updateHessianMatrixSym(bool part, cv::Mat &alpha, cv::Mat &beta,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat faces, std::vector<cv::Mat> ims,std::vector<std::vector<int> > lmInds, std::vector<cv::Mat> landIms, BFMSymParams &params, cv::Mat &exprW, bool show ){
	int M = alpha.rows;
	int EM = exprW.rows;
	int nTri = 450;
	float step;
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	params.hessDiag.release();
	params.hessDiag = cv::Mat::zeros(2*M+EM+RENDER_PARAMS_COUNT,1,CV_32F);
	//printf("sno_step --------\n"); 
	//std::cout << "alpha " << alpha.t() <<std::endl;
	//std::cout << "beta " << beta.t() <<std::endl;
	//printf("renderParams ");
	//for (int i=0;i<RENDER_PARAMS_COUNT;i++)
	//	printf("(%d) %f, ",i,renderParams[i]);
	//printf("\n");
	cv::Mat alpha2, beta2, expr2;
	cv::Mat centerPoints[2], centerTexs, normals[2];
	cv::Mat ccenterPoints[2], ccenterTexs, cnormals[2];
	float renderParams2[RENDER_PARAMS_COUNT];

	cv::Mat texEdge3D[2],conEdge3D[2],texEdge3D2[2], conEdge3D2[2];
	std::vector<cv::Point2f> pPoints[2];

	//double time = (double)cv::getTickCount();
	cES = eS(alpha, beta, params);
	float currEF = eFSym(part, alpha, lmInds, landIms,r3Ds, t3Ds, renderParams, exprW); 
	cEF = currEF;
	cETE = cECE = 0;
	if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
		if (part) {
			texEdge3D[0] = festimator.getTriByAlphaParts(alpha,params.texEdgeVisIndices[0],exprW);
			texEdge3D[1] = festimator.getTriByAlphaPartsFlipExpr(alpha,params.texEdgeVisIndices[1],exprW);
		}
		else {
			texEdge3D[0] = festimator.getTriByAlpha(alpha,params.texEdgeVisIndices[0],exprW);
			texEdge3D[1] = festimator.getTriByAlphaFlipExpr(alpha,params.texEdgeVisIndices[1],exprW);
		}
		projectPoints(texEdge3D[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
		projectPoints(texEdge3D[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
		cETE = eESym(ims[0],pPoints,params,0);
	}
	if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
		if (part) {
			conEdge3D[0] = festimator.getTriByAlphaParts(alpha,params.conEdgeIndices[0],exprW);
			conEdge3D[1] = festimator.getTriByAlphaPartsFlipExpr(alpha,params.conEdgeIndices[1],exprW);
		}
		else {
			conEdge3D[0] = festimator.getTriByAlpha(alpha,params.conEdgeIndices[0],exprW);
			conEdge3D[1] = festimator.getTriByAlphaFlipExpr(alpha,params.conEdgeIndices[1],exprW);
		}
		projectPoints(conEdge3D[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
		projectPoints(conEdge3D[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
		cECE = eESym(ims[0],pPoints,params,1);
	}
	//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
	//std::cout << "Times passed EF: " << time << std::endl;

	//time = (double)cv::getTickCount();
	std::vector<int> inds;
	randSelectTrianglesSym(nTri, params, inds);
	//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
	//std::cout << "Times passed randSelect: " << time << std::endl;
	//for (int i=0;i< 30;i++)
	//	printf("%d ",inds[i]);
	//printf("\n ");
	//double time = (double)cv::getTickCount();
	if (params.computeEI) {
		getTrianglesCenterNormalSym(part, alpha, beta,  faces, inds, centerPoints, centerTexs, normals, exprW);
		ccenterPoints[0] = centerPoints[0].clone();
		ccenterPoints[1] = centerPoints[1].clone();
		ccenterTexs = centerTexs.clone();
		cnormals[0] = normals[0].clone();
		cnormals[1] = normals[1].clone();
	}
	float currEI = eISym(ims, centerPoints, centerTexs, normals,r3Ds, t3Ds,renderParams,inds,params,exprW);
	//printf("hesscurrEF %f %f, %f %f\n",currEF,currEI,cETE, cECE);
	//time = ((double)cv::getTickCount() - time)/cv::getTickFrequency(); 
	//std::cout << "Times passed EI: " << time << std::endl;
	//write_SelectedTri("tmp_tri.ply", alpha, faces,inds, centerPoints, centerTexs, normals);
	//getchar();
	// alpha
	step = mstep*20;
	//printf("alpha \n");
	if (params.optimizeAB[0]) {
		for (int i=0;i<M; i++){
			alpha2.release(); alpha2 = alpha.clone();
			alpha2.at<float>(i,0) += step;
			float cES2 = eS(alpha2, beta, params);
			float tmpEF1 = eFSym(part, alpha2, lmInds, landIms, r3Ds, t3Ds, renderParams,exprW);
			//printf("tmpEF1 %f\n",tmpEF1);

			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) {
					texEdge3D2[0] = festimator.getTriByAlphaParts(alpha2,params.texEdgeVisIndices[0],exprW);
					texEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha2,params.texEdgeVisIndices[1],exprW);
				}
				else {
					texEdge3D2[0] = festimator.getTriByAlpha(alpha2,params.texEdgeVisIndices[0],exprW);
					texEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha2,params.texEdgeVisIndices[1],exprW);
				}
				projectPoints(texEdge3D2[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D2[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) {
					conEdge3D2[0] = festimator.getTriByAlphaParts(alpha2,params.conEdgeIndices[0],exprW);
					conEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha2,params.conEdgeIndices[1],exprW);
				}
				else {
					conEdge3D2[0] = festimator.getTriByAlpha(alpha2,params.conEdgeIndices[0],exprW);
					conEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha2,params.conEdgeIndices[1],exprW);
				}
				projectPoints(conEdge3D2[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D2[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE += cECE2 - cECE;
			}
			if (params.computeEI)
				getTrianglesCenterVNormalSym(part, alpha2, faces, inds, centerPoints, normals,exprW);
			float tmpEI1 = eISym(ims, centerPoints, ccenterTexs, normals,r3Ds, t3Ds, renderParams,inds,params,exprW);
			//printf("tmpEI1 %f\n",tmpEI1);
			alpha2.at<float>(i,0) -= 2*step;
			float cES3 = eS(alpha2, beta, params);
			float tmpEF2 = eFSym(part, alpha2, lmInds, landIms,r3Ds, t3Ds, renderParams,exprW);
			//printf("tmpEF2 %f\n",tmpEF2);
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) {
					texEdge3D2[0] = festimator.getTriByAlphaParts(alpha2,params.texEdgeVisIndices[0],exprW);
					texEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha2,params.texEdgeVisIndices[1],exprW);
				}
				else {
					texEdge3D2[0] = festimator.getTriByAlpha(alpha2,params.texEdgeVisIndices[0],exprW);
					texEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha2,params.texEdgeVisIndices[1],exprW);
				}
				projectPoints(texEdge3D2[0],r3Ds,t3Ds,k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D2[1],r3Ds,t3Ds,k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) {
					conEdge3D2[0] = festimator.getTriByAlphaParts(alpha2,params.conEdgeIndices[0],exprW);
					conEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha2,params.conEdgeIndices[1],exprW);
				}
				else {
					conEdge3D2[0] = festimator.getTriByAlpha(alpha2,params.conEdgeIndices[0],exprW);
					conEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha2,params.conEdgeIndices[1],exprW);
				}
				projectPoints(conEdge3D2[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D2[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE += cECE2 - cECE;
			}

			if (params.computeEI)
				getTrianglesCenterVNormalSym(part, alpha2,  faces, inds, centerPoints, normals,exprW);
			float tmpEI2 = eISym(ims, centerPoints, ccenterTexs, normals,r3Ds,t3Ds,renderParams,inds,params, exprW);
			//printf("tmpEI2 %f\n",tmpEI2);
			params.hessDiag.at<float>(i,0) = (params.sI * (tmpEI1 - 2*currEI + tmpEI2) + params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2))/(step*step) 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/(step*step) 
				+ 2/(0.25f*M) + cES3+cES2 - 2*cES;
		}
	}
	// beta
	step = mstep*20;
	if (params.optimizeAB[1]) {
		for (int i=0;i<M; i++){
			beta2.release(); beta2 = beta.clone();
			beta2.at<float>(i,0) += step;
			float cES2 = eS(alpha, beta2, params);
			if (params.computeEI)
				getTrianglesCenterTex(part, beta2,  faces, inds, centerTexs);
			float tmpEI1 = eISym(ims, ccenterPoints, centerTexs, cnormals,r3Ds, t3Ds,renderParams,inds,params,exprW);
			//printf("tmpEI1 %f\n",tmpEI1);
			beta2.at<float>(i,0) -= 2*step;
			float cES3 = eS(alpha, beta2, params);
			if (params.computeEI)
				getTrianglesCenterTex(part, beta2,  faces, inds, centerTexs);
			float tmpEI2 = eISym(ims, ccenterPoints, centerTexs, cnormals,r3Ds, t3Ds,renderParams,inds,params,exprW);
			//printf("tmpEI2 %f\n",tmpEI2);
			params.hessDiag.at<float>(M+i,0) = (params.sI * (tmpEI1 - 2*currEI + tmpEI2))/(step*step) + 2/(0.5f*M) + cES3 + cES2 -2*cES;
			//params.hessDiag.at<float>(M+i,0) = 0;
		}
	}
	// expr
	step = mstep*5;
	if (params.optimizeExpr) {
		for (int i=0;i<EM; i++){
			expr2.release(); expr2 = exprW.clone();
			expr2.at<float>(i,0) += step;
			float tmpEF1 = eFSym(part, alpha, lmInds, landIms, r3Ds, t3Ds,renderParams,expr2);

			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) {
					texEdge3D2[0] = festimator.getTriByAlphaParts(alpha,params.texEdgeVisIndices[0],expr2);
					texEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha,params.texEdgeVisIndices[1],expr2);
				}
				else {
					texEdge3D2[0] = festimator.getTriByAlpha(alpha,params.texEdgeVisIndices[0],expr2);
					texEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha,params.texEdgeVisIndices[1],expr2);
				}
				projectPoints(texEdge3D2[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D2[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) {
					conEdge3D2[0] = festimator.getTriByAlphaParts(alpha,params.conEdgeIndices[0],expr2);
					conEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha,params.conEdgeIndices[1],expr2);
				}
				else {
					conEdge3D2[0] = festimator.getTriByAlpha(alpha,params.conEdgeIndices[0],expr2);
					conEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha,params.conEdgeIndices[1],expr2);
				}
				projectPoints(conEdge3D2[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D2[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE += cECE2 - cECE;
			}
			if (params.computeEI)
				getTrianglesCenterVNormalSym(part, alpha, faces, inds, centerPoints, normals,expr2);
			float tmpEI1 = eISym(ims, centerPoints, ccenterTexs, normals,r3Ds,t3Ds,renderParams,inds,params,expr2);
			//printf("tmpEI1 %f\n",tmpEI1);
			expr2.at<float>(i,0) -= 2*step;
			float tmpEF2 = eFSym(part, alpha, lmInds, landIms, r3Ds, t3Ds, renderParams,expr2);
			//printf("tmpEF2 %f\n",tmpEF2);
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				if (part) {
					texEdge3D2[0] = festimator.getTriByAlphaParts(alpha,params.texEdgeVisIndices[0],expr2);
					texEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha,params.texEdgeVisIndices[1],expr2);
				}
				else {
					texEdge3D2[0] = festimator.getTriByAlpha(alpha,params.texEdgeVisIndices[0],expr2);
					texEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha,params.texEdgeVisIndices[1],expr2);
				}
				projectPoints(texEdge3D2[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D2[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				if (part) {
					conEdge3D2[0] = festimator.getTriByAlphaParts(alpha,params.conEdgeIndices[0],expr2);
					conEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha,params.conEdgeIndices[1],expr2);
				}
				else {
					conEdge3D2[0] = festimator.getTriByAlpha(alpha,params.conEdgeIndices[0],expr2);
					conEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha,params.conEdgeIndices[1],expr2);
				}
				projectPoints(conEdge3D2[0],r3Ds[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D2[1],r3Ds[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE += cECE2 - cECE;
			}

			if (params.computeEI)
				getTrianglesCenterVNormalSym(part, alpha,  faces, inds, centerPoints, normals,expr2);
			float tmpEI2 = eISym(ims, centerPoints, ccenterTexs, normals,r3Ds, t3Ds, renderParams,inds,params,expr2);
			params.hessDiag.at<float>(2*M+i,0) = (params.sI * (tmpEI1 - 2*currEI + tmpEI2) + params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2))/(step*step) 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/(step*step) 
				+ params.sExpr * 2/(0.25f*29) ;
			//printf("tmpEI %f %f <> %f -> %f\n",params.sI, (tmpEI1 - 2*currEI + tmpEI2), params.sI * (tmpEI1 - 2*currEI + tmpEI2)/ (step*step), params.hessDiag.at<float>(2*M+i,0));
		}
	}
	// r
	//step = 0.05;
	step = mstep*2;
	//step = 0.02;
	//step = 0.01;
	if (params.doOptimize[RENDER_PARAMS_R]) {
		std::vector<cv::Mat> r3Ds_2;
		for (int k=0;k<2; k++){
			r3Ds_2.push_back(r3Ds[k].clone());
		}
		for (int i=0;i<3; i++){
			r3Ds_2[0].at<float>(i,0) += step;
			if (i == 1 || i==2) r3Ds_2[1].at<float>(i,0) = -r3Ds_2[0].at<float>(i,0);
			else r3Ds_2[1].at<float>(i,0) = r3Ds_2[0].at<float>(i,0);
			//memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			//renderParams2[RENDER_PARAMS_R+i] += step;
			float tmpEF1 = eFSym(part, alpha, lmInds, landIms, r3Ds_2, t3Ds, renderParams,exprW);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D[0],r3Ds_2[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D[1],r3Ds_2[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D[0],r3Ds_2[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D[1],r3Ds_2[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE += cECE2 - cECE;
			}
			//printf("tmpEF1 %f\n",tmpEF1);
			float tmpEI1 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds_2,t3Ds,renderParams,inds,params, exprW);
			//printf("tmpEI1 %f\n",tmpEI1);
			//renderParams2[RENDER_PARAMS_R+i] -= 2*step;
			r3Ds_2[0].at<float>(i,0) -= 2*step;
			if (i == 1 || i==2) r3Ds_2[1].at<float>(i,0) = -r3Ds_2[0].at<float>(i,0);
			else r3Ds_2[1].at<float>(i,0) = r3Ds_2[0].at<float>(i,0);
			float tmpEF2 = eFSym(part, alpha, lmInds, landIms,r3Ds_2,t3Ds, renderParams,exprW);
			//printf("tmpEF2 %f\n",tmpEF2);
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D[0],r3Ds_2[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D[1],r3Ds_2[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D[0],r3Ds_2[0],t3Ds[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D[1],r3Ds_2[1],t3Ds[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE += cECE2 - cECE;
			}

			float tmpEI2 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds_2,t3Ds,renderParams,inds,params, exprW);
			//printf("tmpEI2 %f\n",tmpEI2);
			params.hessDiag.at<float>(2*M+EM+i,0) = (params.sI * (tmpEI1 - 2*currEI + tmpEI2) + params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2))/(step*step) 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/(step*step) 
				+ 2.0f/params.sR[RENDER_PARAMS_R+i];
			r3Ds_2[0].at<float>(i,0) = r3Ds[0].at<float>(i,0);
			r3Ds_2[1].at<float>(i,0) = r3Ds[1].at<float>(i,0);
		}
	}
	// t
	step = mstep*10;
	//step = 0.05;
	//step = 0.1;
	if (params.doOptimize[RENDER_PARAMS_T]) {
		std::vector<cv::Mat> t3Ds_2;
		for (int k=0;k<2; k++){
			t3Ds_2.push_back(t3Ds[k].clone());
		}
		for (int i=0;i<3; i++){
			t3Ds_2[0].at<float>(i,0) += step;
			if (i == 0) t3Ds_2[1].at<float>(i,0) = -t3Ds_2[0].at<float>(i,0);
			else t3Ds_2[1].at<float>(i,0) = t3Ds_2[0].at<float>(i,0);
			//memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			//renderParams2[RENDER_PARAMS_T+i] += step;
			float tmpEF1 = eFSym(part, alpha, lmInds, landIms, r3Ds, t3Ds_2, renderParams,exprW);
			float dTE = 0;
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D[0],r3Ds[0], t3Ds_2[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D[1],r3Ds[1], t3Ds_2[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			float dCE = 0;
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D[0],r3Ds[0], t3Ds_2[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D[1],r3Ds[1], t3Ds_2[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE += cECE2 - cECE;
			}
			float tmpEI1 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds_2,renderParams,inds,params, exprW);
			//renderParams2[RENDER_PARAMS_T+i] -= 2*step;
			t3Ds_2[0].at<float>(i,0) -= 2*step;
			if (i == 0) t3Ds_2[1].at<float>(i,0) = -t3Ds_2[0].at<float>(i,0);
			else t3Ds_2[1].at<float>(i,0) = t3Ds_2[0].at<float>(i,0);
			float tmpEF2 = eFSym(part, alpha, lmInds, landIms,r3Ds, t3Ds_2, renderParams,exprW);
			if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
				projectPoints(texEdge3D[0],r3Ds[0], t3Ds_2[0],k_m,distCoef,pPoints[0]);
				projectPoints(texEdge3D[1],r3Ds[1], t3Ds_2[1],k_m,distCoef,pPoints[1]);
				float cETE2 = eESym(ims[0],pPoints,params,0);
				dTE += cETE2 - cETE;
			}
			if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
				projectPoints(conEdge3D[0],r3Ds[0], t3Ds_2[0],k_m,distCoef,pPoints[0]);
				projectPoints(conEdge3D[1],r3Ds[1], t3Ds_2[1],k_m,distCoef,pPoints[1]);
				float cECE2 = eESym(ims[0],pPoints,params,1);
				dCE += cECE2 - cECE;
			}
			float tmpEI2 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds_2,renderParams,inds,params, exprW);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_T+i,0) = (params.sI * (tmpEI1 - 2*currEI + tmpEI2) + params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2))/(step*step) 
				+ (params.sF[FEATURES_TEXTURE_EDGE] * dTE + params.sF[FEATURES_CONTOUR_EDGE] * dCE)/(step*step) 
				+ 2.0f/params.sR[RENDER_PARAMS_T+i];
			t3Ds_2[0].at<float>(i,0) = t3Ds[0].at<float>(i,0);
			t3Ds_2[1].at<float>(i,0) = t3Ds[1].at<float>(i,0);
		}
	}
	// AMBIENT
	step = mstep;
	if (params.doOptimize[RENDER_PARAMS_AMBIENT]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_AMBIENT+i] += step;
			float tmpEI1 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds, renderParams2,inds,params, exprW);
			renderParams2[RENDER_PARAMS_AMBIENT+i] -= 2*step;
			float tmpEI2 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds,renderParams2,inds,params, exprW);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_AMBIENT+i,0)  = (params.sI * (tmpEI1 - 2*currEI + tmpEI2))/(step*step) + 2.0f/params.sR[RENDER_PARAMS_AMBIENT+i];
			//params.hessDiag.at<float>(2*M+RENDER_PARAMS_AMBIENT+i,0)  = 0;
		}
	}
	// DIFFUSE
	//step = 0.02;
	if (params.doOptimize[RENDER_PARAMS_DIFFUSE]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_DIFFUSE+i] += step;
			float tmpEI1 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds,renderParams2,inds,params,exprW);
			renderParams2[RENDER_PARAMS_DIFFUSE+i] -= 2*step;
			float tmpEI2 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds,renderParams2,inds,params,exprW);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_DIFFUSE+i,0)  = (params.sI * (tmpEI1 - 2*currEI + tmpEI2))/(step*step) + 2.0f/params.sR[RENDER_PARAMS_DIFFUSE+i];
			//params.hessDiag.at<float>(2*M+RENDER_PARAMS_DIFFUSE+i,0)  = 0;
		}
	}
	// LDIR
	//step = 0.02;
	step = 0.01;
	if (params.doOptimize[RENDER_PARAMS_LDIR]) {
		for (int i=0;i<2; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_LDIR+i] += step;
			float tmpEI1 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds, renderParams2,inds,params, exprW);
			renderParams2[RENDER_PARAMS_LDIR+i] -= 2*step;
			float tmpEI2 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds, t3Ds, renderParams2,inds,params, exprW);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_LDIR+i,0)  = (params.sI * (tmpEI1 - 2*currEI + tmpEI2))/(step*step) + 2.0f/params.sR[RENDER_PARAMS_LDIR+i];
			//params.hessDiag.at<float>(2*M+RENDER_PARAMS_LDIR+i,0)  = 0;
		}
	}
	// others
	//step = 0.01;
	step = mstep;
	for (int i=RENDER_PARAMS_CONTRAST;i<RENDER_PARAMS_COUNT; i++){
		if (params.doOptimize[i]) {
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[i] += step;
			float tmpEI1 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds,t3Ds,renderParams2,inds,params,exprW);
			renderParams2[i] -= 2*step;
			float tmpEI2 = eISym(ims, ccenterPoints, ccenterTexs, cnormals,r3Ds,t3Ds,renderParams2,inds,params,exprW);
			params.hessDiag.at<float>(2*M+EM+i,0)  = (params.sI * (tmpEI1 - 2*currEI + tmpEI2))/(step*step) + 2.0f/params.sR[i];
			//params.hessDiag.at<float>(2*M+i,0)  = 0;
		}
	}
	ccenterPoints[0].release(); ccenterTexs.release();	cnormals[0].release();
	ccenterPoints[1].release(); cnormals[1].release();
	return currEI;
}

void FaceServices2::renderFaceSym(char* fname, cv::Mat colorIm, bool part, cv::Mat alpha, cv::Mat beta,cv::Mat faces,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat exprW ){
	Mat k_m(3,3,CV_32F,_k);
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	//RenderServices rs;
	cv::Mat shape[2];
	cv::Mat tex;
	float renderParams2[RENDER_PARAMS_COUNT];
	memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);
	if (!part) {
		shape[0] = festimator.getShape(alpha,exprW);
		shape[1] = festimator.getShapeFlipExpr(alpha,exprW);
		tex = festimator.getTexture(beta);
	}
	else {
		shape[0] = festimator.getShapeParts(alpha,exprW);
		shape[1] = festimator.getShapePartsFlipExpr(alpha,exprW);
		tex = festimator.getTextureParts(beta);
	}

	cv::Mat vecR(3,1,CV_32F), vecT(3,1,CV_32F);
	cv::Mat colors;
	cv::Mat trgA(3,1,CV_32F);
	trgA.at<float>(0,0) = 0.0f;
	trgA.at<float>(1,0) = 0.0f;
	trgA.at<float>(2,0) = 1.0f;
	cv::Mat vecL(3,1,CV_32F);
	vecL.at<float>(0,0) = cos(renderParams[RENDER_PARAMS_LDIR])*sin(renderParams[RENDER_PARAMS_LDIR+1]);
	vecL.at<float>(1,0) = sin(renderParams[RENDER_PARAMS_LDIR]);
	vecL.at<float>(2,0) = cos(renderParams[RENDER_PARAMS_LDIR])*cos(renderParams[RENDER_PARAMS_LDIR+1]);
	cv::Mat vecL_r(3,1,CV_32F);
	vecL_r.at<float>(0,0) = -cos(renderParams[RENDER_PARAMS_LDIR])*sin(renderParams[RENDER_PARAMS_LDIR+1]);
	vecL_r.at<float>(1,0) = sin(renderParams[RENDER_PARAMS_LDIR]);
	vecL_r.at<float>(2,0) = cos(renderParams[RENDER_PARAMS_LDIR])*cos(renderParams[RENDER_PARAMS_LDIR+1]);
	cv::Mat matR = findRotation(vecL,trgA);
	cv::Mat matR_r = findRotation(vecL_r,trgA);
	float r2[3];
	float t2[3];
	t2[0] = t2[1] = 0.00001;
	cv::Mat vecR2(3,1,CV_32F,r2);

	int K = r3Ds.size();

	cv::Mat refRGB, refDepth;
	cv::Mat refRGB2, refDepth2;

	bool* visible = new bool[shape[0].rows];
	bool* noShadow = new bool[shape[0].rows];
	float* r = renderParams2 + RENDER_PARAMS_R;
	float* t = renderParams2 + RENDER_PARAMS_T;
	for (int k=0;k<K;k++) {
		refRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
		refDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);

		for (int i=0;i<3;i++){
			r[i] = r3Ds[k].at<float>(i,0);
			t[i] = t3Ds[k].at<float>(i,0);
			vecR.at<float>(i,0) = r[i];
			vecT.at<float>(i,0) = t[i];
		}
		t2[2] = t[2] * 1.5;
		//printf("render %d: %f %f %f, %f %f %f\n",k,r[0],r[1],r[2],t[0],t[1],t[2]);
		im_render->copyShape(shape[k]);
		im_render->loadModel();
		im_render->render(r,t,_k[4],refRGB,refDepth);
		projectCheckVis(im_render, shape[k], r, t, refDepth, visible);

		cv::Mat matR1;
		cv::Rodrigues(vecR,matR1);
		cv::Mat matR2;
		if (k<K/2) matR2 = matR*matR1;
		else matR2 = matR_r*matR1;

		cv::Rodrigues(matR2,vecR2);

		refRGB2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
		refDepth2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
		im_render->render(r2,t2,_k[4],refRGB2,refDepth2);
		projectCheckVis(im_render, shape[k], r2, t2, refDepth2, noShadow);

		if (k>=K/2) renderParams2[RENDER_PARAMS_LDIR+1] = - renderParams[RENDER_PARAMS_LDIR+1];
		rs.estimateColor(shape[k],tex,faces,visible,noShadow,renderParams2,colors);
		im_render->copyColors(colors);
		im_render->loadModel();
		refRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
		refDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
		im_render->render(r,t,_k[4],refRGB,refDepth);
		//if (cenP.rows > 0) {
		//	vector<Point2f> projPoints;
		//	cv::projectPoints( cenP, vecR, vecT, k_m, distCoef, projPoints );
		//	for (int pp = 0;pp<projPoints.size();pp++) {
		//		cv::circle(refRGB,Point(projPoints[pp].x,projPoints[pp].y),1,Scalar(0,0,255),1);
		//	}
		//}

		//for (int i=0;i<landIm.rows;i++){
		//	cv::circle(refRGB,Point(landIm.at<float>(i,0),landIm.at<float>(i,1)),1,Scalar(255,0,0),1);
		//}
		//printf("render %d %f %f %f, %f %f %f\n",k,r[0],r[1],r[2],t[0],t[1],t[2]);
		if (k>=K/2) {
			cv::flip(refRGB,refRGB2,1);
			//refRGB = refRGB2.clone();
			refRGB = cv::Mat::zeros(refRGB2.rows,refRGB2.cols,refRGB2.type());
			refRGB2(cv::Rect(0,0,refRGB2.cols-1,refRGB2.rows)).copyTo(refRGB(cv::Rect(1,0,refRGB2.cols-1,refRGB2.rows)));
			//refRGB2(cv::Rect(1,0,refRGB2.cols-1,refRGB2.rows)).copyTo(refRGB(cv::Rect(0,0,refRGB2.cols-1,refRGB2.rows)));
		}

		char fname2[200];
		sprintf(fname2,"%s_%04d.png",fname,k);
		imwrite(fname2,refRGB);
		//im_render->render(r2,t2,_k[4],refRGB2,refDepth2);
		//sprintf(fname2,"l_%s_%04d.png",fname,k);
		//imwrite(fname2,refRGB2);
	}

	shape[0].release(); shape[1].release(); tex.release(); refRGB.release(); refDepth.release();
	refRGB2.release(); refDepth2.release();
	delete visible; delete noShadow;
}

bool FaceServices2::updateTrianglesSym(cv::Mat colorIm,cv::Mat faces,bool part, cv::Mat alpha,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, BFMSymParams &params, cv::Mat exprW ){
	Mat k_m(3,3,CV_32F,_k);
	//RenderServices rs;
	cv::Mat shape[2];
	if (!part) {
		shape[0] = festimator.getShape(alpha,exprW);
		shape[1] = festimator.getShapeFlipExpr(alpha,exprW);
	}
	else {
		shape[0] = festimator.getShapeParts(alpha,exprW); 
		shape[1] = festimator.getShapePartsFlipExpr(alpha,exprW); 
	}

	cv::Mat vecR(3,1,CV_32F), vecT(3,1,CV_32F);
	cv::Mat colors;

	cv::Mat refRGB, refDepth;
	cv::Mat refRGB2, refDepth2;

	bool* visible = new bool[shape[0].rows];
	bool* noShadow = new bool[shape[0].rows];

	float* r = renderParams + RENDER_PARAMS_R;
	float* t = renderParams + RENDER_PARAMS_T;

	cv::Mat trgA(3,1,CV_32F);
	trgA.at<float>(0,0) = 0.0f;
	trgA.at<float>(1,0) = 0.0f;
	trgA.at<float>(2,0) = 1.0f;
	cv::Mat vecL(3,1,CV_32F);
	vecL.at<float>(0,0) = cos(renderParams[RENDER_PARAMS_LDIR])*sin(renderParams[RENDER_PARAMS_LDIR+1]);
	vecL.at<float>(1,0) = sin(renderParams[RENDER_PARAMS_LDIR]);
	vecL.at<float>(2,0) = cos(renderParams[RENDER_PARAMS_LDIR])*cos(renderParams[RENDER_PARAMS_LDIR+1]);
	cv::Mat matR = findRotation(vecL,trgA);
	cv::Mat vecL_r(3,1,CV_32F);
	vecL_r.at<float>(0,0) = -cos(renderParams[RENDER_PARAMS_LDIR])*sin(renderParams[RENDER_PARAMS_LDIR+1]);
	vecL_r.at<float>(1,0) = sin(renderParams[RENDER_PARAMS_LDIR]);
	vecL_r.at<float>(2,0) = cos(renderParams[RENDER_PARAMS_LDIR])*cos(renderParams[RENDER_PARAMS_LDIR+1]);
	cv::Mat matR_r = findRotation(vecL_r,trgA);

	float r2[3];
	float t2[3];
	t2[0] = t2[1] = 0.00001;
	t2[2] = t[2] * 1.5;
	cv::Mat vecR2(3,1,CV_32F,r2);
	float sumArea = 0;

	int K = r3Ds.size();
	for (int k=0;k<r3Ds.size();k++){
		refRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
		refDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
		for (int i=0;i<3;i++){
			r[i] = r3Ds[k].at<float>(i,0);
			t[i] = t3Ds[k].at<float>(i,0);
			vecR.at<float>(i,0) = r[i];
			vecT.at<float>(i,0) = t[i];
		}
		im_render->copyShape(shape[k]);
		im_render->loadModel();
		im_render->render(r,t,_k[4],refRGB,refDepth);
		vector<Point2f> projPoints = projectCheckVis2(im_render, shape[k], r, t, refDepth, visible);

		cv::Mat gradX, gradY, edgeAngles;
		float angleThresh = 2*M_PI/TEXEDGE_ORIENTATION_REG_NUM;
		if (params.sF[FEATURES_TEXTURE_EDGE] > 0 || params.sF[FEATURES_CONTOUR_EDGE] > 0) {
			cv::Mat grayIm, tmpIm;
			cv::cvtColor( refRGB, tmpIm, CV_BGR2GRAY );
			cv::blur( tmpIm, grayIm, cv::Size(3,3) );
			cv::Scharr(grayIm,gradX,CV_32F,1,0);
			cv::Scharr(grayIm,gradY,CV_32F,0,1);
			//cv::imshow("grayIm",grayIm);
			edgeAngles = cv::Mat::zeros(grayIm.rows,grayIm.cols,CV_32F);
		}

		if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
			params.texEdgeVisIndices[k].clear();
			params.texEdgeVisBin[k].clear();

			for (int i=0;i<texEdgeIndices.size();i++) {
				if (visible[texEdgeIndices[i]])
					params.texEdgeVisIndices[k].push_back(texEdgeIndices[i]);
			}

			for (int i=0;i<params.texEdgeVisIndices[k].size();i++) {
				int ix = floor(projPoints.at(params.texEdgeVisIndices[k][i]).x+0.5);
				int iy = floor(projPoints.at(params.texEdgeVisIndices[k][i]).y+0.5);
				if (k == 1) ix = colorIm.cols /*+ 1*/ - ix;
				//cv::circle(tmppIm,cv::Point(ix,iy),1,cv::Scalar(0,0,255));
				if (ix < 0 || iy < 0 || ix > colorIm.cols-1 || iy > colorIm.rows-1)
					params.texEdgeVisBin[k].push_back(-1);
				else {
					float ange;
					//if (k < 1) 
						ange = atan2(gradY.at<float>(iy,ix),gradX.at<float>(iy,ix));
					//else ange = atan2(gradY.at<float>(iy,ix),-gradX.at<float>(iy,ix));
					int bin = floor((ange + M_PI)/angleThresh + 0.5);
					if (bin >= TEXEDGE_ORIENTATION_REG_NUM) bin = bin - TEXEDGE_ORIENTATION_REG_NUM;
					params.texEdgeVisBin[k].push_back(bin);
				}
			}
			//cv::imwrite("tex.png",tmppIm);
			//getchar();

			//cv::Mat vColors = cv::Mat::zeros(shape.rows,3,CV_32F);
			//for (int i=0;i<params.texEdgeVisIndices.size();i++){
			//	int ind = params.texEdgeVisIndices[i];
			//	vColors.at<float>(ind,0) = vColors.at<float>(ind,1) = vColors.at<float>(ind,2) = 255;
			//}
			//write_plyFloat("edges.ply",shape,vColors,faces);

			//imshow("conMap",refRGB*255);
			//cv::waitKey();
		}

		if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
			params.conEdgeIndices[k].clear();
			params.conEdgeBin[k].clear();

			cv::Mat vertexMap, dMap, binMap, tmpIm, conMap;
			vertexMap = -cv::Mat::ones(colorIm.rows,colorIm.cols,CV_32S);
			dMap = 100*cv::Mat::ones(colorIm.rows,colorIm.cols,CV_32F);
			for (int i=0;i<projPoints.size();i++) {
				if (visible[i]){
					int ix = floor(projPoints[i].x+0.5);
					int iy = floor(projPoints[i].y+0.5);
					if (k == 1) ix = colorIm.cols/* + 1*/ - ix;
					if (ix >= 0 && iy >= 0 && ix < colorIm.cols && iy < colorIm.rows) {
						float dist = sqrt(pow(projPoints[i].x-ix,2) + pow(projPoints[i].y - iy,2));
						if (dist < dMap.at<float>(iy,ix)){
							vertexMap.at<int>(iy,ix) = i;
							dMap.at<float>(iy,ix) = dist;
						}
					}
				}
			}
			binMap = refDepth < 0.9999;
			binMap.convertTo(binMap,CV_8U);

			Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 3, 3 ), Point( 1, 1 ) );
			erode( binMap, tmpIm, element );
			conMap = binMap - tmpIm;

			for (int i=0;i<conMap.rows;i++) {
				for (int j=0;j<conMap.cols;j++) {
					if (conMap.at<unsigned char>(i,j) > 0 && vertexMap.at<int>(i,j) >= 0 && BaselFace::BaselFace_canContour[vertexMap.at<int>(i,j)]){
						int ind = vertexMap.at<int>(i,j);
						params.conEdgeIndices[k].push_back(ind);
						float ange = atan2(gradY.at<float>(i,j),gradX.at<float>(i,j));
						int bin = floor((ange + M_PI)/angleThresh + 0.5);
						bin = bin % (TEXEDGE_ORIENTATION_REG_NUM/2);
						params.conEdgeBin[k].push_back(bin);
					}
				}
			}
		}

			cv::Mat matR1;
			cv::Rodrigues(vecR,matR1);
			cv::Mat matR2;
			if (k < K/2) matR2 = matR*matR1;
			else matR2 = matR_r*matR1;

			cv::Rodrigues(matR2,vecR2);

			refRGB2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
			refDepth2 = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
			im_render->render(r2,t2,_k[4],refRGB2,refDepth2);
			projectCheckVis(im_render, shape[k], r2, t2, refDepth2, noShadow);
			//memset(noShadow,true, sizeof(bool)*im_render->face_->mesh_.nVertices_);

			params.triVis[k].release(); params.triNoShadow[k].release();
			params.triVis[k] = cv::Mat::zeros(faces.rows,1,CV_8U);
			params.triNoShadow[k] = cv::Mat::zeros(faces.rows,1,CV_8U);
			for (int i=0;i<faces.rows;i++){
				unsigned int val = 1;
				if (BaselFace::BaselFace_keepV[i] == 0) val = 0;
				else {
					for (int j=0;j<3;j++) {
						if (!visible[faces.at<int>(i,j)]) {
							val = 0; break;
						}
					}
				}
				params.triVis[k].at<unsigned int>(i,0) = val;
				if (val) {
					for (int j=0;j<3;j++) {
						if (!noShadow[faces.at<int>(i,j)]) {
							val = 0; break;
						}
					}
					params.triNoShadow[k].at<unsigned int>(i,0) = val;
				}
			}

			params.triAreas[k].release();
			params.triAreas[k] = cv::Mat::zeros(faces.rows,1,CV_32F);
			for (int i=0;i<faces.rows;i++){
				if (params.triVis[k].at<unsigned int>(i,0)) {
					Point2f a = projPoints[faces.at<int>(i,0)];
					Point2f b = projPoints[faces.at<int>(i,1)];
					Point2f c = projPoints[faces.at<int>(i,2)];
					Point2f v1 = b - a;
					Point2f v2 = c - a;
					params.triAreas[k].at<float>(i,0) = abs(v1.x*v2.y - v1.y*v2.x);
					sumArea += params.triAreas[k].at<float>(i,0);
				}
			}
		}
		params.triCSAreas.release();
		params.triCSAreas = cv::Mat::zeros(faces.rows,1,CV_32F);
		params.trisumAreas.release();
		params.trisumAreas = cv::Mat::zeros(faces.rows,1,CV_32F);
		float cssum = 0;
		int bin = 0;
		int new_bin;
		int prev_i = 0;
		for (int i=0;i<faces.rows;i++){
			params.triCSAreas.at<float>(i,0) = cssum;
			float ss = 0;
			for (int k=0;k<r3Ds.size();k++){
				params.triAreas[k].at<float>(i,0) /= sumArea;
				ss += params.triAreas[k].at<float>(i,0);
			}
			params.trisumAreas.at<float>(i,0) = ss; 
			cssum += ss;
			new_bin = floor(cssum * NUM_AREA_BIN);
			if (new_bin > bin) {
				for (int j=bin;j<new_bin;j++)
					params.indexCSArea[j] = prev_i;
				bin = new_bin;
				prev_i = i;
			}
		}
		for (;bin<NUM_AREA_BIN;bin++)
			params.indexCSArea[bin] = prev_i;
		//printf("************ indexing\n");
		//for (int i=0;i<50;i++)
		//	printf("(%d = %d) %f >= %f;  \n",i,params.indexCSArea[i],params.triCSAreas.at<float>(params.indexCSArea[i],0), ((float)i)/NUM_AREA_BIN);
		//getchar();

		shape[0].release();shape[1].release();
		refRGB.release();refDepth.release();
		refRGB2.release();refDepth2.release();
		delete visible; delete noShadow;
		return true;
}

float FaceServices2::eISym(std::vector<cv::Mat> &ims,cv::Mat* centerPointsArr, cv::Mat centerTexs, cv::Mat* normalsArr,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams,std::vector<int> inds, BFMSymParams &params, cv::Mat &exprW, bool show) {	
	if (!params.computeEI) return 0;
	float renderParams1[RENDER_PARAMS_COUNT];
	float renderParams2[RENDER_PARAMS_COUNT];
	//float* storeErr = new float[centerPointsArr[0].rows];
	memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
	memcpy(renderParams1,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
	renderParams2[RENDER_PARAMS_LDIR+1] = - renderParams2[RENDER_PARAMS_LDIR+1];
	//printf("multiEI\n");
	char text[200];
	Mat k_m(3,3,CV_32F,_k);
	cv::Mat colors;
	float err = 0;
	int numPoints = 0;
	//RenderServices rs;
	//printf("start EI\n");
	cv::Mat colorIm2;
	int K = ims.size();
	for (int k=0;k<ims.size();k++) {
		for (int i=0;i<3;i++){
			renderParams1[RENDER_PARAMS_R+i] = renderParams2[RENDER_PARAMS_R+i] = r3Ds[k].at<float>(i,0);
			renderParams1[RENDER_PARAMS_T+i] = renderParams2[RENDER_PARAMS_T+i] = t3Ds[k].at<float>(i,0);
		}
		if (k<K/2) rs.estimatePointColor(centerPointsArr[k], centerTexs, normalsArr[k], inds, params.triVis[k], params.triNoShadow[k], renderParams1, colors);
		else rs.estimatePointColor(centerPointsArr[k], centerTexs, normalsArr[k], inds, params.triVis[k], params.triNoShadow[k], renderParams2, colors);
		std::vector<cv::Point2f> allImgPts;
		cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	
		//printf("projectPoints\n");
		cv::projectPoints( centerPointsArr[k], r3Ds[k], t3Ds[k], k_m, distCoef, allImgPts );
		if (show) colorIm2 = ims[k].clone();
		int snumPoints = 0;
		float sErr = 0;
		for (int i=0;i<centerPointsArr[k].rows;i++){
			int ind = inds[i];
			if (params.triVis[k].at<unsigned char>(ind,0) == 0) continue;
			//if (allImgPts[i].x < 0 || allImgPts[i].x >= colorIm.cols-1 || allImgPts[i].y < 0 || allImgPts[i].y >= colorIm.rows-1)
			//	printf("outer point\n");
			CvPoint2D64f tmpPoint = cvPoint2D64f(allImgPts[i].x,allImgPts[i].y);
			cv::Vec3d imcolor = avSubMatValue8UC3_2( &tmpPoint, &(ims[k]) );
			if (show) cv::circle(colorIm2,cv::Point(allImgPts[i].x,allImgPts[i].y),1,cv::Scalar(colors.at<float>(i,2),colors.at<float>(i,1),colors.at<float>(i,0)),2);
			//cv::Vec3b imcolor = colorIm.at<Vec3b>(allImgPts[i].y,allImgPts[i].x);
			float sval = 0;
			for (int j=0;j<3;j++){
				float val = (colors.at<float>(i,j) - imcolor(2-j))/5.0f;
				if (abs(val) < 4) 
					sval += val*val;
				else
					sval += 2*4*abs(val) - 4*4;
			}
			//storeErr[snumPoints] = sqrt(sval);
			sErr += sval;
			snumPoints++;
		}
		//float avgErr = sqrt(sErr/snumPoints);
		//err += avgErr;

		err += sErr;
		numPoints += snumPoints;
		//for (int i=0;i<snumPoints;i++){
		//	if (!params.optimizeAB[1] || (avgErr - storeErr[i]) < 5) {
		//		err += storeErr[i]*storeErr[i];
		//		numPoints++;
		//	}
		//}
		//if (show) {
		//	printf("%d: %d\n",k,numPoints);
		//	sprintf(text,"pp_%04d.png",k);
		//	cv::imwrite(text,colorIm2);
		//}
	}
	//delete storeErr;
	//printf("end EI %f\n",sqrt(err/numPoints));
	//if (show) getchar();
	return sqrt(err/numPoints);
}

float FaceServices2::eFSym(bool part, cv::Mat alpha, std::vector<std::vector<int> > inds, std::vector<cv::Mat> landIms,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat &exprW, bool show){
	Mat k_m(3,3,CV_32F,_k);
	//printf("%f\n",renderParams[RENDER_PARAMS_R+1]);
	cv::Mat mLM[2];
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	char text[200];
	//if (part) {
	//	write_plyFloat("vismLM.ply",mLM.t());
	//	getchar();
	//}
	
	//if (show)
	//printf("pose (%f - %f, %f - %f, %f - %f) (%f - %f, %f - %f, %f -%f)\n",r3Ds[0].at<float>(0,0), r3Ds[1].at<float>(0,0)
	//	,r3Ds[0].at<float>(1,0), r3Ds[1].at<float>(1,0),r3Ds[0].at<float>(2,0), r3Ds[1].at<float>(2,0)
	//	,t3Ds[0].at<float>(0,0), t3Ds[1].at<float>(0,0),t3Ds[0].at<float>(1,0), t3Ds[1].at<float>(1,0)
	//	,t3Ds[0].at<float>(2,0), t3Ds[1].at<float>(2,0));
	float err = 0;
	//cv::Mat colorIm2;
	for (int k=0;k<landIms.size();k++){
		//colorIm2 = cv::Mat::zeros((int)(_k[5]*2),(int)(_k[2]*2),CV_8UC3);
		//for (int i=0;i<3;i++){
		//	renderParams[RENDER_PARAMS_R+i] = r3Ds[k].at<float>(i,0);
		//	renderParams[RENDER_PARAMS_T+i] = t3Ds[k].at<float>(i,0);
		//}
		//if (alpha.rows == 99) return 0;
		//mLM.release();
		if (!part) {
			if (k == 0) mLM[k] = festimator.getLMByAlpha(alpha,-r3Ds[k].at<float>(1,0), inds[k],exprW);
			else mLM[k] = festimator.getLMByAlphaFlipExpr(alpha,-r3Ds[k].at<float>(1,0), inds[k],exprW);
		}
		else {
			if (k == 0) mLM[k] = festimator.getLMByAlphaParts(alpha,-r3Ds[k].at<float>(1,0), inds[k],exprW);
			else mLM[k] = festimator.getLMByAlphaPartsFlipExpr(alpha,-r3Ds[k].at<float>(1,0), inds[k],exprW);
		}
		int numPoints = 0;
		float serr = 0;
		//cv::Mat rVec(3,1,CV_32F, renderParams + RENDER_PARAMS_R);
		//cv::Mat tVec(3,1,CV_32F, renderParams + RENDER_PARAMS_T);
		std::vector<cv::Point2f> allImgPts;

		cv::projectPoints( mLM[k], r3Ds[k], t3Ds[k], k_m, distCoef, allImgPts );
		for (int i=0;i<mLM[k].rows;i++){
			//cv::circle(colorIm2,cv::Point(allImgPts[i].x,allImgPts[i].y),1,cv::Scalar(255,0,0),2);
			//cv::circle(colorIm2,cv::Point(landIms[k].at<float>(i,0),landIms[k].at<float>(i,1)),1,cv::Scalar(0,0,255),2);
			float val = landIms[k].at<float>(i,0) - allImgPts[i].x;
			//if (val > 4) val = 4;
			//if (val < -4) val = -4;
			serr += val*val;
			val = landIms[k].at<float>(i,1) - allImgPts[i].y;
			//if (val > 4) val = 4;
			//if (val < -4) val = -4;
			serr += val*val;
			numPoints++;
		}
		if (show) printf("serr %d %f (%d)\n",k,serr,numPoints);
		err += sqrt(serr/numPoints);
		//sprintf(text,"ff%04d.png",k);
		//cv::imwrite(text,colorIm2);
	}
	
	return err/landIms.size();
}

float FaceServices2::line_searchSym(bool part, cv::Mat &alpha, cv::Mat &beta,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat &dirMove,std::vector<int> inds, cv::Mat faces, std::vector<cv::Mat> colorIms,std::vector<std::vector<int> > lmInds, std::vector<cv::Mat> landIms, BFMSymParams &params, cv::Mat &exprW, int maxIters ){
	float step = 1.0f;
	float sstep = 2.0f;
	float minStep = 0.00001f;
	//float cCost = computeCost(cEF, cEI,cETE, cECE, alpha, beta, renderParams,params);
	cv::Mat alpha2, beta2, exprW2;
	float renderParams2[RENDER_PARAMS_COUNT];
	alpha2 = alpha.clone();
	beta2 = beta.clone();
	exprW2 = exprW.clone();
	std::vector<cv::Mat> r3Ds_2, t3Ds_2;
	for (int k=0;k<2; k++){
		r3Ds_2.push_back(r3Ds[k].clone());
		t3Ds_2.push_back(t3Ds[k].clone());
	}
	memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);

	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	cv::Mat texEdge3D2[2], conEdge3D2[2];
	std::vector<cv::Point2f> pPoints[2];

	int M = alpha.rows;
	int EM = exprW.rows;
	float ssize = 0;
	for (int i=0;i<dirMove.rows;i++) ssize += dirMove.at<float>(i,0)*dirMove.at<float>(i,0);
	ssize = sqrt(ssize);
	//printf("ssize: %f vs %f\n",ssize,(2*M+EM+RENDER_PARAMS_COUNT)/160.0f);
	if (ssize > (2*M+EM+RENDER_PARAMS_COUNT)/160.0f) {
		step = (2*M+EM+RENDER_PARAMS_COUNT)/(160.0f * ssize);
		ssize = (2*M+EM+RENDER_PARAMS_COUNT)/160.0f;
	}
	if (ssize < minStep){
		return 0;
	}
	int tstep = floor(log(ssize/minStep));
	if (tstep < maxIters) maxIters = tstep;

	cv::Mat centerPoints[2], centerTexs, normals[2];
	for (int i=0;i<3;i++) {
		renderParams[RENDER_PARAMS_R+i] = r3Ds[0].at<float>(i,0);
		renderParams[RENDER_PARAMS_T+i] = t3Ds[0].at<float>(i,0);
	}
	float curCost = computeCostSym(cEF, cEI,cETE, cECE,cES, alpha, beta, r3Ds, t3Ds, renderParams, params, exprW );
	//printf("curCost %f\n",curCost);

	bool hasNoBound = false;
	int iter = 0;
	for (; iter<maxIters; iter++){
		// update
		if (params.optimizeAB[0]){
			for (int i=0;i<M;i++) {
				float tmp = alpha.at<float>(i,0) + step*dirMove.at<float>(i,0);
				if (tmp >= maxVal) alpha2.at<float>(i,0) = maxVal;
				else if (tmp <= -maxVal) alpha2.at<float>(i,0) = -maxVal;
				else {
					alpha2.at<float>(i,0) = tmp;
					hasNoBound = true;
				}
			}
		}
		if (params.optimizeAB[1]){
			for (int i=0;i<M;i++) {
				float tmp = beta.at<float>(i,0) + step*dirMove.at<float>(M+i,0);
				if (tmp >= maxVal) beta2.at<float>(i,0) = maxVal;
				else if (tmp <= -maxVal) beta2.at<float>(i,0) = -maxVal;
				else {
					beta2.at<float>(i,0) = tmp;
					hasNoBound = true;
				}
			}
		}
		if (params.optimizeExpr){
			for (int i=0;i<EM;i++) {
				float tmp = exprW.at<float>(i,0) + step*dirMove.at<float>(2*M+i,0);
				if (tmp >= 3) exprW2.at<float>(i,0) = 3;
				else if (tmp <= -3) exprW2.at<float>(i,0) = -3;
				else {
					exprW2.at<float>(i,0) = tmp;
					hasNoBound = true;
				}
			}
		}

		if (params.doOptimize[RENDER_PARAMS_R]){
			for (int i=0;i<3;i++) {
				renderParams2[RENDER_PARAMS_R + i] = r3Ds[0].at<float>(i,0) + step*dirMove.at<float>(2*M+EM+RENDER_PARAMS_R+i,0);
				r3Ds_2[0].at<float>(i,0) = renderParams2[RENDER_PARAMS_R + i];
				if (i == 1 || i==2) r3Ds_2[1].at<float>(i,0) = -renderParams2[RENDER_PARAMS_R + i];
				else r3Ds_2[1].at<float>(i,0) = renderParams2[RENDER_PARAMS_R + i];
			}
		}
		
		if (params.doOptimize[RENDER_PARAMS_T]){
			for (int i=0;i<3;i++) {
				renderParams2[RENDER_PARAMS_T + i] = t3Ds[0].at<float>(i,0) + step*dirMove.at<float>(2*M+EM+RENDER_PARAMS_T+i,0);
				t3Ds_2[0].at<float>(i,0) = renderParams2[RENDER_PARAMS_T + i];
				if (i == 0) t3Ds_2[1].at<float>(i,0) = -renderParams2[RENDER_PARAMS_T + i];
				else t3Ds_2[1].at<float>(i,0) = renderParams2[RENDER_PARAMS_T + i];
			}
		}
		//printf("Update pose (%f %f %f) (%f %f %f)\n", r3Ds_2[0].at<float>(0,0), r3Ds_2[0].at<float>(1,0), r3Ds_2[0].at<float>(2,0)
		//	, t3Ds_2[0].at<float>(0,0), t3Ds_2[0].at<float>(1,0), t3Ds_2[0].at<float>(2,0));

		for (int i=6;i<RENDER_PARAMS_COUNT;i++) {
			if (params.doOptimize[i]){
				float tmp = renderParams[i] + step*dirMove.at<float>(2*M+EM+i,0);
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (tmp > 1.0) renderParams2[i] = 1.0f;
					else if (tmp < -1.0) renderParams2[i] = -1.0f;
					else {
						renderParams2[i] = tmp;
						hasNoBound = true;
					}
				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (tmp >= 3.0) renderParams2[i] = 3.0f;
					else if (tmp <= -3.0) renderParams2[i] = -3.0f;
					else {
						renderParams2[i] = tmp;
						hasNoBound = true;
					}
				}
				else renderParams2[i] = tmp;
			}
		}
		if (!hasNoBound) {
			iter = maxIters; break;
		}
		float tmpEF = cEF;
		if (params.sF[FEATURES_LANDMARK] > 0) tmpEF = eFSym(part,alpha2, lmInds,landIms,r3Ds_2, t3Ds_2,renderParams2, exprW2);
		float tmpEI = cEI;
		if (params.sI > 0 && params.computeEI) {
			getTrianglesCenterNormalSym(part, alpha2,beta2,  faces, inds, centerPoints,centerTexs, normals, exprW2);
			tmpEI = eISym(colorIms,centerPoints,centerTexs, normals,r3Ds_2, t3Ds_2,renderParams2,inds,params, exprW2);
		}
		float tmpETE = cETE;
		if (params.sF[FEATURES_TEXTURE_EDGE] > 0) {
			if (part) {
				texEdge3D2[0] = festimator.getTriByAlphaParts(alpha2,params.texEdgeVisIndices[0], exprW2);
				texEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha2,params.texEdgeVisIndices[1], exprW2);
			}
			else {
				texEdge3D2[0] = festimator.getTriByAlpha(alpha2,params.texEdgeVisIndices[0],exprW2);
				texEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha2,params.texEdgeVisIndices[1],exprW2);
			}
			projectPoints(texEdge3D2[0],r3Ds_2[0], t3Ds_2[0],k_m,distCoef,pPoints[0]);
			projectPoints(texEdge3D2[1],r3Ds_2[1], t3Ds_2[1],k_m,distCoef,pPoints[1]);
			tmpETE = eESym(colorIms[0],pPoints,params,0);
		}
		float tmpECE = cECE;
		if (params.sF[FEATURES_CONTOUR_EDGE] > 0) {
			if (part) {
				conEdge3D2[0] = festimator.getTriByAlphaParts(alpha2,params.conEdgeIndices[0],exprW2);
				conEdge3D2[1] = festimator.getTriByAlphaPartsFlipExpr(alpha2,params.conEdgeIndices[1],exprW2);
			}
			else {
				conEdge3D2[0] = festimator.getTriByAlpha(alpha2,params.conEdgeIndices[0],exprW2);
				conEdge3D2[1] = festimator.getTriByAlphaFlipExpr(alpha2,params.conEdgeIndices[1],exprW2);
			}
			projectPoints(conEdge3D2[0],r3Ds_2[0],t3Ds_2[0],k_m,distCoef,pPoints[0]);
			projectPoints(conEdge3D2[1],r3Ds_2[1],t3Ds_2[1],k_m,distCoef,pPoints[1]);
			float tmpECE = eESym(colorIms[0],pPoints,params,1);
		}

		float tmpES = eS(alpha2, beta2, params);
		float tmpCost = computeCostSym(tmpEF, tmpEI,tmpETE, tmpECE, tmpES, alpha2, beta2, r3Ds_2, t3Ds_2, renderParams2, params,exprW2 );
		//printf("tmpCost %f (%f)\n",tmpCost,step);
		if (tmpCost < curCost) {
			break;
		}
		else {
			step = step/sstep;
			//printf("step %f\n",step);
		}
	}
	//getchar();
	if (iter >= maxIters) return 0;
	else return step;
}

float FaceServices2::computeCostSym(float vEF, float vEI,float vETE, float vECE,float vS, cv::Mat &alpha, cv::Mat &beta,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, BFMSymParams &params, cv::Mat &exprW ){
	float val = params.sF[FEATURES_LANDMARK]*vEF + params.sI*vEI + params.sF[FEATURES_TEXTURE_EDGE]*vETE + params.sF[FEATURES_CONTOUR_EDGE]*vECE + vS;
	int M = alpha.rows;
	//printf("cost (%f) = %f ",vEF, val); 
	if (params.optimizeAB[0]){
		for (int i=0;i<M;i++)
			val += alpha.at<float>(i,0)*alpha.at<float>(i,0)/(0.25f*M);
	}
	if (params.optimizeAB[1]){
		for (int i=0;i<M;i++)
			val += beta.at<float>(i,0)*beta.at<float>(i,0)/(0.5f*M);
	}
	//printf("-> %f ",val); 
	if (params.optimizeExpr){
		for (int i=0;i<exprW.rows;i++)
			val += params.sExpr * exprW.at<float>(i,0)*exprW.at<float>(i,0)/(0.5f*29);
	}
	//printf("-> %f ",val); 
	for (int i=0;i<3;i++) {
		if (params.doOptimize[RENDER_PARAMS_R]){
			val += (r3Ds[0].at<float>(i,0) - params.initR[RENDER_PARAMS_R+i])*(r3Ds[0].at<float>(i,0) - params.initR[RENDER_PARAMS_R+i])/params.sR[RENDER_PARAMS_R+i];
		}
	}
	for (int i=0;i<3;i++) {
		if (params.doOptimize[RENDER_PARAMS_T]){
			val += (t3Ds[0].at<float>(i,0) - params.initR[RENDER_PARAMS_T+i])*(t3Ds[0].at<float>(i,0) - params.initR[RENDER_PARAMS_T+i])/params.sR[RENDER_PARAMS_T+i];
		}
	}
	for (int i=6;i<RENDER_PARAMS_COUNT;i++) {
		if (params.doOptimize[i]){
			val += (renderParams[i] - params.initR[i])*(renderParams[i] - params.initR[i])/params.sR[i];
		}
	}
	//printf("-> %f ",val); 
	return val;
}
float FaceServices2::eESym(cv::Mat colorIm, std::vector<cv::Point2f> *projPoints, BFMSymParams &params, int type){
	if ((type == 1 && params.sF[FEATURES_CONTOUR_EDGE] == 0) || (type == 0 && params.sF[FEATURES_TEXTURE_EDGE] == 0)) return 0;
	float err = 0;
	int nPoints = 0;
	for (int k=0;k<2;k++) {
		for (int i=0;i<projPoints[k].size();i++){
			int ix = floor(projPoints[k][i].x+0.5);
			int iy = floor(projPoints[k][i].y+0.5);
			if (k == 1) {
				ix = colorIm.cols/* + 1*/ - ix;
			}
			if (ix >= 0 && iy >= 0 && ix < colorIm.cols && iy < colorIm.rows) {
				if (type == 0) {		// Texture edges
					int bin = params.texEdgeVisBin[k][i];
					CvPoint2D64f tmpPoint = cvPoint2D64f(projPoints[k][i].x,projPoints[k][i].y);
					float val = avSubMatValue32F( &tmpPoint, texEdgeDist + bin );
					err += val*val;
					nPoints++;
				}
				else {
					int bin = params.conEdgeBin[k][i];
					CvPoint2D64f tmpPoint = cvPoint2D64f(projPoints[k][i].x,projPoints[k][i].y);
					float val = avSubMatValue32F( &tmpPoint, conEdgeDist + bin );
					err += val*val;
					nPoints++;
				}
			}
		}
	}
	return sqrt(err/nPoints);
}

void FaceServices2::getTrianglesCenterNormalSym(bool part, cv::Mat alpha,cv::Mat beta, cv::Mat faces,std::vector<int> inds0, cv::Mat* centerPointsArr, cv::Mat &centerTexs, cv::Mat* normalsArr, cv::Mat exprW){
	//printf("getTrianglesCenterNormal\n");
	//if (centerPointsArr == 0) centerPointsArr = new cv::Mat[2];
	//if (normalsArr == 0) normalsArr = new cv::Mat[2];
	std::vector<int> inds1 = verticesFromTriangles(faces,inds0);
	std::vector<int> inds2 = aVerticesFromTriangles(faces,inds0);

	cv::Mat vertices1, vertices2, texs; 
	if (!part) {
		vertices1 = festimator.getTriByAlpha(alpha,inds1,exprW);
		vertices2 = festimator.getTriByAlphaFlipExpr(alpha,inds1,exprW);
		texs = festimator.getTriByBeta(beta,inds2);
	}
	else {
		vertices1 = festimator.getTriByAlphaParts(alpha,inds1,exprW);
		vertices2 = festimator.getTriByAlphaPartsFlipExpr(alpha,inds1,exprW);
		texs = festimator.getTriByBetaParts(beta,inds2);
	}
	if (centerTexs.cols == 0) centerTexs = cv::Mat::zeros(inds0.size(),3,CV_32F);
	if (centerPointsArr[0].cols == 0) centerPointsArr[0] = cv::Mat::zeros(inds0.size(),3,CV_32F);
	if (normalsArr[0].cols == 0) normalsArr[0] = cv::Mat::zeros(inds0.size(),3,CV_32F);
	for (int i=0;i<inds0.size();i++){
		Point3f a(vertices1.at<float>(3*i,0),vertices1.at<float>(3*i,1),vertices1.at<float>(3*i,2));
		Point3f b(vertices1.at<float>(3*i+1,0),vertices1.at<float>(3*i+1,1),vertices1.at<float>(3*i+1,2));
		Point3f c(vertices1.at<float>(3*i+2,0),vertices1.at<float>(3*i+2,1),vertices1.at<float>(3*i+2,2));
		Point3f v1 = b-a;
		Point3f v2 = c-a;
		//Point3f v0 = a+b+c;
		Point3f v = v1.cross(v2);
		float normm = sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
		//centerPoints.at<float>(i,0) = v0.x/3;
		//centerPoints.at<float>(i,1) = v0.y/3;
		//centerPoints.at<float>(i,2) = v0.z/3;
		centerPointsArr[0].at<float>(i,0) = a.x;
		centerPointsArr[0].at<float>(i,1) = a.y;
		centerPointsArr[0].at<float>(i,2) = a.z;
		normalsArr[0].at<float>(i,0) = v.x/normm;
		normalsArr[0].at<float>(i,1) = v.y/normm;
		normalsArr[0].at<float>(i,2) = v.z/normm;
	}
	if (centerPointsArr[1].cols == 0) centerPointsArr[1] = cv::Mat::zeros(inds0.size(),3,CV_32F);
	if (normalsArr[1].cols == 0) normalsArr[1] = cv::Mat::zeros(inds0.size(),3,CV_32F);
	for (int i=0;i<inds0.size();i++){
		Point3f a(vertices2.at<float>(3*i,0),vertices2.at<float>(3*i,1),vertices2.at<float>(3*i,2));
		Point3f b(vertices2.at<float>(3*i+1,0),vertices2.at<float>(3*i+1,1),vertices2.at<float>(3*i+1,2));
		Point3f c(vertices2.at<float>(3*i+2,0),vertices2.at<float>(3*i+2,1),vertices2.at<float>(3*i+2,2));
		Point3f v1 = b-a;
		Point3f v2 = c-a;
		//Point3f v0 = a+b+c;
		Point3f v = v1.cross(v2);
		float normm = sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
		//centerPoints.at<float>(i,0) = v0.x/3;
		//centerPoints.at<float>(i,1) = v0.y/3;
		//centerPoints.at<float>(i,2) = v0.z/3;
		centerPointsArr[1].at<float>(i,0) = a.x;
		centerPointsArr[1].at<float>(i,1) = a.y;
		centerPointsArr[1].at<float>(i,2) = a.z;
		normalsArr[1].at<float>(i,0) = v.x/normm;
		normalsArr[1].at<float>(i,1) = v.y/normm;
		normalsArr[1].at<float>(i,2) = v.z/normm;
	}
	for (int i=0;i<inds0.size();i++){
		//Point3f a(texs.at<float>(3*i,0),texs.at<float>(3*i,1),texs.at<float>(3*i,2));
		//Point3f b(texs.at<float>(3*i+1,0),texs.at<float>(3*i+1,1),texs.at<float>(3*i+1,2));
		//Point3f c(texs.at<float>(3*i+2,0),texs.at<float>(3*i+2,1),texs.at<float>(3*i+2,2));
		//Point3f v0 = a+b+c;
		centerTexs.at<float>(i,0) = texs.at<float>(i,0);
		centerTexs.at<float>(i,1) = texs.at<float>(i,1);
		centerTexs.at<float>(i,2) = texs.at<float>(i,2);
	}
}

void FaceServices2::getTrianglesCenterVNormalSym(bool part, cv::Mat alpha, cv::Mat faces,std::vector<int> inds0, cv::Mat* centerPointsArr, cv::Mat* normalsArr, cv::Mat exprW){
	//printf("getTrianglesCenterVNormalSym\n");
	//if (centerPointsArr == 0) centerPointsArr = new cv::Mat[2];
	//if (normalsArr == 0) normalsArr = new cv::Mat[2];
	std::vector<int> inds1 = verticesFromTriangles(faces,inds0);
	cv::Mat vertices1, vertices2; 
	if (!part) {
		vertices1 = festimator.getTriByAlpha(alpha,inds1,exprW);
		vertices2 = festimator.getTriByAlphaFlipExpr(alpha,inds1,exprW);
	}
	else {
		vertices1 = festimator.getTriByAlphaParts(alpha,inds1,exprW);
		vertices2 = festimator.getTriByAlphaPartsFlipExpr(alpha,inds1,exprW);
	}
	if (centerPointsArr[0].cols == 0) centerPointsArr[0] = cv::Mat::zeros(inds0.size(),3,CV_32F);
	if (normalsArr[0].cols == 0) normalsArr[0] = cv::Mat::zeros(inds0.size(),3,CV_32F);
	for (int i=0;i<inds0.size();i++){
		Point3f a(vertices1.at<float>(3*i,0),vertices1.at<float>(3*i,1),vertices1.at<float>(3*i,2));
		Point3f b(vertices1.at<float>(3*i+1,0),vertices1.at<float>(3*i+1,1),vertices1.at<float>(3*i+1,2));
		Point3f c(vertices1.at<float>(3*i+2,0),vertices1.at<float>(3*i+2,1),vertices1.at<float>(3*i+2,2));
		Point3f v1 = b-a;
		Point3f v2 = c-a;
		//Point3f v0 = a+b+c;
		Point3f v = v1.cross(v2);
		float normm = sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
		//centerPoints.at<float>(i,0) = v0.x/3;
		//centerPoints.at<float>(i,1) = v0.y/3;
		//centerPoints.at<float>(i,2) = v0.z/3;
		centerPointsArr[0].at<float>(i,0) = a.x;
		centerPointsArr[0].at<float>(i,1) = a.y;
		centerPointsArr[0].at<float>(i,2) = a.z;
		normalsArr[0].at<float>(i,0) = v.x/normm;
		normalsArr[0].at<float>(i,1) = v.y/normm;
		normalsArr[0].at<float>(i,2) = v.z/normm;
	}
	if (centerPointsArr[1].cols == 0) centerPointsArr[1] = cv::Mat::zeros(inds0.size(),3,CV_32F);
	if (normalsArr[1].cols == 0) normalsArr[1] = cv::Mat::zeros(inds0.size(),3,CV_32F);
	for (int i=0;i<inds0.size();i++){
		Point3f a(vertices2.at<float>(3*i,0),vertices2.at<float>(3*i,1),vertices2.at<float>(3*i,2));
		Point3f b(vertices2.at<float>(3*i+1,0),vertices2.at<float>(3*i+1,1),vertices2.at<float>(3*i+1,2));
		Point3f c(vertices2.at<float>(3*i+2,0),vertices2.at<float>(3*i+2,1),vertices2.at<float>(3*i+2,2));
		Point3f v1 = b-a;
		Point3f v2 = c-a;
		//Point3f v0 = a+b+c;
		Point3f v = v1.cross(v2);
		float normm = sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
		//centerPoints.at<float>(i,0) = v0.x/3;
		//centerPoints.at<float>(i,1) = v0.y/3;
		//centerPoints.at<float>(i,2) = v0.z/3;
		centerPointsArr[1].at<float>(i,0) = a.x;
		centerPointsArr[1].at<float>(i,1) = a.y;
		centerPointsArr[1].at<float>(i,2) = a.z;
		normalsArr[1].at<float>(i,0) = v.x/normm;
		normalsArr[1].at<float>(i,1) = v.y/normm;
		normalsArr[1].at<float>(i,2) = v.z/normm;
	}
}
bool FaceServices2::sfs(cv::Mat colorIm, cv::Mat lms,cv::Vec6d poseCLM, float conf,cv::Mat lmVis, cv::Mat &shape, cv::Mat &tex, string model_file, string lm_file, string pose_file, string refDir){
	float renderParams[RENDER_PARAMS_COUNT];
	float renderParams2[RENDER_PARAMS_COUNT];
	//nTri = 40;
	//nTriHess = 300;
	Mat k_m(3,3,CV_32F,_k);
	//BaselFaceEstimator festimator;
	BFMParams params;
	double time = (double)cv::getTickCount();
	params.init();
	prepareEdgeDistanceMaps(colorIm);
	//time = (double)cv::getTickCount();
	Mat alpha = cv::Mat::zeros(20,1,CV_32F);
	Mat beta = cv::Mat::zeros(20,1,CV_32F);
	Mat exprW = cv::Mat::zeros(29,1,CV_32F);
	Mat alpha_bk, beta_bk, exprW_bk;
	shape = festimator.getShape(alpha);
	tex = festimator.getTexture(beta);
	Mat landModel0 = festimator.getLM(shape,poseCLM(4));
	float bCost, cCost, fCost;
	int bestIter = 0;
	bCost = 10000.0f;
	//write_plyFloat("visLM0.ply",landModel0.t());
	std::vector<int> lmVisInd;
	for (int i=0;i<60;i++){
		if (lmVis.at<int>(i)){
			if (/*(i< 17 || i> 26) &&*/ (i > 16 || abs(poseCLM(4)) <= M_PI/10 || (poseCLM(4) > M_PI/10 && i > 7) || (poseCLM(4) < -M_PI/10 && i < 9)))
				lmVisInd.push_back(i);
		}
	}
	cv::Mat tmpIm = colorIm.clone();

	if (lmVisInd.size() < 8) return false;
	Mat landModel = cv::Mat( lmVisInd.size(),3,CV_32F);
	Mat landIm = cv::Mat( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landModel.at<float>(i,0) = landModel0.at<float>(ind,0);
		landModel.at<float>(i,1) = landModel0.at<float>(ind,1);
		landModel.at<float>(i,2) = landModel0.at<float>(ind,2);
		landIm.at<float>(i,0) = lms.at<double>(ind);
		landIm.at<float>(i,1) = lms.at<double>(ind+landModel0.rows);
		//cv::circle(tmpIm,Point(landIm.at<float>(i,0),landIm.at<float>(i,1)),1,Scalar(255,0,0),1);
	}
	//imwrite("visLM.png",tmpIm);
	//write_plyFloat("visLM.ply",landModel.t());
	//getchar();

	cv::Mat vecR, vecT;
	festimator.estimatePose3D(landModel,landIm,k_m,vecR,vecT);
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_R+i] = vecR.at<float>(i,0);
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_T+i] = vecT.at<float>(i,0);
	memcpy(renderParams,params.initR,sizeof(float)*RENDER_PARAMS_COUNT);

	cv::Mat faces = festimator.getFaces() - 1;
	cv::Mat faces_fill = festimator.getFaces_fill() - 1;
	cv::Mat colors;

	memset(params.sF,0,sizeof(float)*NUM_EXTRA_FEATURES);

	params.sI = 0.0;
	params.sF[FEATURES_LANDMARK] = 8.0f;
	//params.sF[FEATURES_TEXTURE_EDGE] = 4.0f;
	char text[200];
	Mat alpha0, beta0;
	int iter=0;
	int badCount = 0;
	int M = 99;
	int NEI = 1;
	//params.sF[FEATURES_TEXTURE_EDGE] = 6;
	//params.sF[FEATURES_CONTOUR_EDGE] = 6;
	//params.optimizeAB[0] = params.optimizeAB[1] = false;
	memset(params.doOptimize,true,sizeof(bool)*6);

	int EM = 29;
	M = 99;
	float renderParams_tmp[RENDER_PARAMS_COUNT];
	cv::Mat exprW_tmp = cv::Mat::zeros(29,1,CV_32F);
	loadReference(refDir, model_file, alpha, beta, renderParams, M, exprW, EM);
	std::cout << alpha << std::endl;
	//float maxside = 200;
	//if (colorIm.cols > maxside || colorIm.rows > maxside){
	//	float maxedge = (colorIm.cols > colorIm.rows)?colorIm.cols:colorIm.rows;
	//	float rescale = maxedge/maxside;
	//	int newH = floor(colorIm.rows/rescale);
	//	int newW = floor(colorIm.cols/rescale);
	//	cv::Mat colorIm2 = colorIm.clone();
	//	cv::resize(colorIm2,colorIm,cv::Size(newW,newH));
	//	_k[0] /= rescale;
	//	_k[4] /= rescale;
	//	_k[2] = newW/2.0f;
	//	_k[5] = newW/2.0f;
	//	printf("New size: %d %d\n",newH,newW);
	//}
	
	im_render = new FImRenderer(cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3));
	im_render->loadMesh(shape,tex,faces_fill);
	updateTriangles(colorIm,faces,false,  alpha, renderParams, params, exprW );

	//cv::Mat tmpRender = renderFace(text, colorIm,landIm,false,  alpha, beta, faces, renderParams, exprW ).clone();
	shape = festimator.getShape(alpha,exprW);
	tex = festimator.getTexture(beta);


	im_render->copyShape(shape);
	im_render->copyColors(tex);

	cv::Mat refRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	cv::Mat refDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);

	float r[3], t[3];
	for (int i=0;i<3;i++){
		r[i] = renderParams[RENDER_PARAMS_R+i];
		t[i] = renderParams[RENDER_PARAMS_T+i];
	}
	im_render->loadModel();
	im_render->render(r,t,_k[4],refRGB,refDepth);

	cv::Mat mask = refDepth < 0.9999;
	imwrite("mask.png",mask);
	float zNear_ = im_render->zNear;
	float zFar_ = im_render->zFar;

	float minDepth = 0;
	std::vector<int> pointInd[2];
	cv::Mat vindex = cv::Mat::zeros(mask.rows,mask.cols,CV_32S)-1;
	int numNeighbors = 0;
	for (int x=0;x<refDepth.cols;x++){
		for (int y=0;y<refDepth.rows;y++){
			float dd = refDepth.at<float>(y,x);
			if (dd<0.9999){
				vindex.at<int>(y,x) = pointInd[0].size();
				pointInd[0].push_back(y);
				pointInd[1].push_back(x);
				refDepth.at<float>(y,x) = - zNear_*zFar_   / ( zFar_ - dd * ( zFar_ - zNear_ ));
				if (minDepth > refDepth.at<float>(y,x))
					minDepth = refDepth.at<float>(y,x);
			}
		}
	}
	std::vector<int>* neighbors =  new std::vector<int>[pointInd[0].size()];
	for (int i=0;i<pointInd[0].size();i++){
		// Check neighbors
		int y = pointInd[0][i];
		int x = pointInd[1][i];
		int sx = (x-NEI>=0)?(x-NEI):0;
		int ex = x<=(refDepth.cols-1-NEI)?(x+NEI):(refDepth.cols-1);
		int sy = (y-NEI>=0)?(y-NEI):0;
		int ey = y<=(refDepth.rows-1-NEI)?(y+NEI):(refDepth.rows-1);
		for (int rx=sx;rx<=ex;rx++)
			for (int ry=sy;ry<=ey;ry++)
				if ((rx != x || ry != y) && mask.at<unsigned char>(ry,rx)>0) 
			{
					numNeighbors++;
					neighbors[i].push_back(vindex.at<int>(ry,rx));
			}
	}

	//FILE* fneighbors = fopen("neighbors.txt","w");
	//for (int i=0;i<pointInd[0].size();i++){
	//	fprintf(fneighbors,"%d %d %d",i,pointInd[1][i],pointInd[0][i]);
	//	for (int j=0;j<neighbors[i].size();j++)
	//		fprintf(fneighbors," %d %d",pointInd[1][neighbors[i][j]],pointInd[0][neighbors[i][j]]);
	//	fprintf(fneighbors,"\n");
	//}
	//fclose(fneighbors);

	mask = mask/255;
	cv::Mat mask2, refDepth2;
	mask.convertTo(mask2,CV_32F);
	refDepth = refDepth.mul(mask2) + minDepth * (1-mask2);
	refDepth2 = refDepth-minDepth;
	//double mn,mx;
	//cv::minMaxIdx(refDepth2,&mn,&mx);
	//printf("%f %f\n",mn,mx);
	//threshold(refDepth2,refDepth2,255,255,2);
	//refDepth2.convertTo(refDepth2,CV_8U);
	imwrite("depth.png",refDepth2);
	cv::Mat normal_im = computeDepthImageNormal(refDepth,mask);
	//cv::Mat ns[3];
	//cv::split(normal_im,ns);
	//imwrite("nx.png",(ns[0]*50)+100);
	//imwrite("ny.png",(ns[1]*50)+100);

	cv::Mat normalMat = cv::Mat::ones(pointInd[0].size(),4,CV_32F);
	cv::Mat imcolorMat[3];
	for (int i=0;i<3;i++)
		imcolorMat[i] = cv::Mat::ones(pointInd[0].size(),1,CV_32F);
	for (int i=0;i<pointInd[0].size();i++){
		cv::Vec3f n = normal_im.at<cv::Vec3f>(pointInd[0][i],pointInd[1][i]);
		normalMat.at<float>(i,0) = n(0);
		normalMat.at<float>(i,1) = n(1);
		normalMat.at<float>(i,2) = n(2);
		cv::Vec3b c = colorIm.at<cv::Vec3b>(pointInd[0][i],pointInd[1][i]);
		imcolorMat[0].at<float>(i,0) = c(0);
		imcolorMat[1].at<float>(i,0) = c(1);
		imcolorMat[2].at<float>(i,0) = c(2);
	}
	cv::Mat shp[3];
	FILE* shFile = fopen("sh.txt","w");
	for (int i=0;i<3;i++){
		printf("%d %d %d, %d %d %d\n",normalMat.rows,normalMat.cols,normalMat.type(),imcolorMat[i].rows,imcolorMat[i].cols,imcolorMat[i].type());
		cv::solve(normalMat,imcolorMat[i],shp[i],DECOMP_SVD);
		fprintf(shFile,"%f %f %f %f\n",shp[i].at<float>(0,0),shp[i].at<float>(1,0),shp[i].at<float>(2,0),shp[i].at<float>(3,0));
	}
	fclose(shFile);

	cv::Mat tmpOut[3];
	cv::Mat outIm = colorIm*0;
	for (int i=0;i<3;i++){
		tmpOut[i] = normalMat*shp[i];
	}
	for (int j=0;j<pointInd[0].size();j++){
		cv::Vec3b c;
		for (int k=0;k<3;k++){
			float val = tmpOut[k].at<float>(j,0);
			val = (val<0)?0:val;
			val = (val<255)?val:255;
			c(k) = val;
		}
		outIm.at<cv::Vec3b>(pointInd[0][j],pointInd[1][j]) = c;
	}
	imwrite("shMap.png",outIm);

	float theta = 0.05*255*255;
	float delta_c_2 = 0.05*255*255;
	//float theta = 0.25*255*255;
	//float delta_c_2 = 0.25*255*255;
	float delta_d_2 = 50;
	float lambda_p = 0.1;
	//float lambda_p = 100;
	//float lambda_p = 10;
	float lambda_b_a = 1;
	float lambda_b_b = 1;
	float lambda_1 = 0.004*255*255*255;
	float lambda_2 = 0.0075*255*255*255;


	IndWeight* neighbors1 =  new IndWeight[pointInd[0].size()*3];
	for (int i=0;i<pointInd[0].size();i++){
		for (int k=0;k<3;k++){
			neighbors1[k*pointInd[0].size()+i].push_back(std::pair<int,double>(i,0));
			for (int j=0;j<neighbors[i].size();j++){
				neighbors1[k*pointInd[0].size()+i].push_back(std::pair<int,double>(neighbors[i][j],0));
			}
		}
	}
	//cv::Mat albedo[3];
	//cv::Mat A_albedo[3], b_albedo[3];
	//for (int i=0;i<3;i++) {
	//	A_albedo[i] = cv::Mat::zeros(numNeighbors+pointInd[0].size(),pointInd[0].size(),CV_32F);
	//}
 
	std::vector<SpT> triA[3];
	Eigen::VectorXd b_albedo[3], albedo[3];
	for (int k=0;k<3;k++) {
		b_albedo[k].resize(pointInd[1].size());
	}
	for (int i=0;i<pointInd[1].size();i++){
		int x = pointInd[1][i];
		int y = pointInd[0][i];
		float dp = refDepth.at<float>(y,x);

		float wA[3] = {0};
		Vec3b c = colorIm.at<Vec3b>(y,x);
		for (int k=0;k<3;k++){
			//float S = tmpOut[k].at<float>(i,0);
			//wA[k] +=S*S;
			double SS = tmpOut[k].at<float>(i,0);
			neighbors1[k*pointInd[0].size()+i].begin()->second += SS * SS;
			b_albedo[k](i) = SS * c(k);
			//trib[k].push_back(SpT(i,0,S*c(k)));
		}
		for (int j=0;j<neighbors[i].size();j++){
			int x1 = pointInd[1][neighbors[i][j]];
			int y1 = pointInd[0][neighbors[i][j]];
			Vec3b c1 = colorIm.at<Vec3b>(y1,x1);
			float dp1 = refDepth.at<float>(y1,x1);
			double w_d = std::exp(-(dp1-dp)*(dp1-dp)/(2*delta_d_2));
			for (int k=0;k<3;k++){
				float idiff = (float)c1(k)-(float)c(k);
				idiff = idiff*idiff;
				if (idiff > theta) continue;
				double w_c = std::exp(-idiff/(2*delta_c_2));
				//double w_c = 1;
				//printf("w_c %f   %f\n",idiff,w_c);
				//triA[k].push_back(SpT(i,neighbors[i][j],-lambda_p*w_c*w_d));
				findByKey(neighbors1,k*pointInd[0].size()+i,neighbors[i][j])->second += -lambda_p*w_c*w_d;
				findByKey(neighbors1,k*pointInd[0].size()+neighbors[i][j],i)->second += lambda_p*w_c*w_d;
				//wA[k] +=lambda_p*w_d*w_c;
				neighbors1[k*pointInd[0].size()+i].begin()->second += lambda_p*w_d*w_c;
			}
		}
		//for (int k=0;k<3;k++){
		//	triA[k].push_back(SpT(i,i,wA[k]));
		//}
	}
	for (int i=0;i<pointInd[0].size();i++){
		for (int k=0;k<3;k++){
			for (IndWeight::iterator it=neighbors1[k*pointInd[0].size()+i].begin();it != neighbors1[k*pointInd[0].size()+i].end();it++){
				if (it->second != 0)
					triA[k].push_back(SpT(i,it->first,it->second));
			}
		}
	}

	for (int k=0;k<3;k++){
		SpMat A_albedo(pointInd[1].size(),pointInd[1].size());
		A_albedo.setFromTriplets(triA[k].begin(),triA[k].end());
		Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
		solver.compute(A_albedo);
		if(solver.info()!=Eigen::Success) {
		  // decomposition failed
		  printf("Solving failed!\n");
		}
		albedo[k] = solver.solve(b_albedo[k]);
	}

	outIm = colorIm*0;
	for (int j=0;j<pointInd[0].size();j++){
		cv::Vec3b c;
		for (int k=0;k<3;k++){
			if (albedo[k](j) < 0) albedo[k](j) = 0;
			if (albedo[k](j) > 2.2) albedo[k](j) = 2.2;
			float val = albedo[k](j) * 120;
			val = (val<0)?0:val;
			val = (val<255)?val:255;
			c(k) = val;
		}
		outIm.at<cv::Vec3b>(pointInd[0][j],pointInd[1][j]) = c;
	}
	imwrite("albedo.png",outIm);
	
	outIm = colorIm*0;
	for (int j=0;j<pointInd[0].size();j++){
		cv::Vec3b c;
		for (int k=0;k<3;k++){
			float val = albedo[k](j) * tmpOut[k].at<float>(j,0);
			val = (val<0)?0:val;
			val = (val<255)?val:255;
			c(k) = val;
		}
		outIm.at<cv::Vec3b>(pointInd[0][j],pointInd[1][j]) = c;
	}
	imwrite("step2.png",outIm);


	// Step 3: beta recover	

	for (int i=0;i<pointInd[0].size();i++){
		for (int k=0;k<3;k++){
			for (IndWeight::iterator it=neighbors1[k*pointInd[0].size()+i].begin();it != neighbors1[k*pointInd[0].size()+i].end();it++){
				it->second = 0;
			}
		}
	}
	Eigen::VectorXd b_beta[3], x_beta[3];
	for (int k=0;k<3;k++) {
		triA[k].clear();
		b_beta[k].resize(pointInd[1].size());
		x_beta[k].resize(pointInd[1].size());
		x_beta[k] = x_beta[k]*0;
	}
	
	for (int i=0;i<pointInd[1].size();i++){
		int x = pointInd[1][i];
		int y = pointInd[0][i];
		float dp = refDepth.at<float>(y,x);

		float wA[3] = {0};
		Vec3b c = colorIm.at<Vec3b>(y,x);
		for (int k=0;k<3;k++){
			float S = tmpOut[k].at<float>(i,0);
			//wA[k] =1+lambda_b_b;
			neighbors1[k*pointInd[1].size()+i].begin()->second += 1+lambda_b_b;
			b_beta[k](i) = (double)c(k) - albedo[k][i]*S;
			//trib[k].push_back(SpT(i,0,S*c(k)));
		}
		for (int j=0;j<neighbors[i].size();j++){
			int x1 = pointInd[1][neighbors[i][j]];
			int y1 = pointInd[0][neighbors[i][j]];
			Vec3b c1 = colorIm.at<Vec3b>(y1,x1);
			float dp1 = refDepth.at<float>(y1,x1);
			double w_d = std::exp(-(dp1-dp)*(dp1-dp)/(2*delta_d_2));
			for (int k=0;k<3;k++){
				float idiff = (float)c1(k)-(float)c(k);
				idiff = idiff*idiff;
				if (idiff > theta) continue;
				double w_c = std::exp(-idiff/(2*delta_c_2));
				//double w_c = 1;
				//printf("w_c %f   %f\n",idiff,w_c);
				////triA[k].push_back(SpT(i,neighbors[i][j],-lambda_b_a*w_c*w_d));
				findByKey(neighbors1,k*pointInd[1].size()+i,neighbors[i][j]) += -lambda_b_a*w_c*w_d;
				findByKey(neighbors1,k*pointInd[1].size()+neighbors[i][j],i) += lambda_b_a*w_c*w_d;
				////wA[k] +=lambda_b_a*w_d*w_c;
				neighbors1[k*pointInd[1].size()+i].begin()->second += lambda_b_a*w_d*w_c;
			}
		}
		//for (int k=0;k<3;k++){
		//	triA[k].push_back(SpT(i,i,wA[k]));
		//}
	}
	for (int i=0;i<pointInd[0].size();i++){
		for (int k=0;k<3;k++){
			for (IndWeight::iterator it=neighbors1[k*pointInd[0].size()+i].begin();it != neighbors1[k*pointInd[0].size()+i].end();it++){
				if (it->second != 0)
					triA[k].push_back(SpT(i,it->first,it->second));
			}
		}
	}
	for (int k=0;k<3;k++){
		SpMat A_beta(pointInd[1].size(),pointInd[1].size());
		A_beta.setFromTriplets(triA[k].begin(),triA[k].end());
		Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
		solver.compute(A_beta);
		if(solver.info()!=Eigen::Success) {
		  // decomposition failed
		  printf("Solving failed!\n");
		}
		x_beta[k] = solver.solve(b_beta[k]);
	}

	outIm = colorIm*0;
	for (int j=0;j<pointInd[0].size();j++){
		cv::Vec3b c;
		for (int k=0;k<3;k++){
			float val = x_beta[k](j) * 20000 + 120;
			val = (val<0)?0:val;
			val = (val<255)?val:255;
			c(k) = val;
		}
		outIm.at<cv::Vec3b>(pointInd[0][j],pointInd[1][j]) = c;
	}
	imwrite("beta.png",outIm);
	
	outIm = colorIm*0;
	for (int j=0;j<pointInd[0].size();j++){
		cv::Vec3b c;
		for (int k=0;k<3;k++){
			float val = x_beta[k](j) + albedo[k](j) * tmpOut[k].at<float>(j,0);
			val = (val<0)?0:val;
			val = (val<255)?val:255;
			c(k) = val;
		}
		outIm.at<cv::Vec3b>(pointInd[0][j],pointInd[1][j]) = c;
	}
	imwrite("step3.png",outIm);

	// Step 4: Update depth
	IndWeight* neighbors2 =  new IndWeight[pointInd[0].size()];
	for (int i=0;i<pointInd[0].size();i++){
		neighbors2[i].push_back(std::pair<int,double>(i,0));
		int x = pointInd[1][i];
		int y = pointInd[0][i];
		for (int rx=x-2;rx<=x+2;rx++){
			if (rx < 0 || rx >= colorIm.cols) continue;
			for (int ry=y-2;ry<=y+2;ry++){
				if (ry < 0 || ry >= colorIm.rows) continue;
				int ddd = abs(rx-x)+abs(ry-y);
				if (ddd> 0 && ddd < 3){
					if (mask.at<unsigned char>(ry,rx) > 0)
						neighbors2[i].push_back(std::pair<int,double>(vindex.at<int>(ry,rx),0));
				}
			}
		}
	}

	refDepth = refDepth-minDepth;
	
		cv::Mat v(pointInd[0].size(),3,CV_32F);
		cv::Mat cl(pointInd[0].size(),3,CV_32F);
		for (int i=0;i<pointInd[0].size();i++){
			v.at<float>(i,2) = refDepth.at<float>(pointInd[0][i],pointInd[1][i])+minDepth;
			v.at<float>(i,0) = v.at<float>(i,2)*(pointInd[1][i] - _k[2])/_k[0];
			v.at<float>(i,1) = v.at<float>(i,2)*(pointInd[0][i] - _k[5])/_k[4];
			cl.at<float>(i,0) = albedo[0](i) * 120;
			cl.at<float>(i,1) = albedo[1](i) * 120;
			cl.at<float>(i,2) = albedo[2](i) * 120;
		}
		std::vector<Vec3i> fac;
		computeFaces(vindex,fac);
		cv::Mat fac2(fac.size(),3,CV_32S);
		for (int i=0;i<fac.size();i++){
			fac2.at<int>(i,0) = fac[i](0);
			fac2.at<int>(i,1) = fac[i](1);
			fac2.at<int>(i,2) = fac[i](2);
		}
		sprintf(text,"%s_initDepth.ply",model_file.c_str());
		write_plyFloat(text,v,cl,fac2);
	for (int diter = 0; diter < 30; diter++){
		triA[0].clear();
		for (int i=0;i<pointInd[0].size();i++){
			for (IndWeight::iterator it=neighbors2[i].begin();it != neighbors2[i].end();it++){
				it->second = 0;
			}
		}
    Eigen::VectorXd b_depth, x_depth;
	b_depth.resize(pointInd[1].size());
		for (int i=0;i<pointInd[1].size();i++){
			int x = pointInd[1][i];
			int y = pointInd[0][i];
			float dp = refDepth.at<float>(y,x);
			Vec3b c = colorIm.at<Vec3b>(y,x);

			double wA = 0;
			double wb = 0;
			//bool sidePoints[4];
			float pp[2] = {0};

			cv::Vec3f n;
			if (x <= 0 || mask.at<unsigned char>(y,x-1) <= 0 || x >= colorIm.cols-1 || mask.at<unsigned char>(y,x+1) <= 0) {
				int dcount=0;
				std::vector<int> members;
					if (y > 0 && mask.at<unsigned char>(y-1,x) > 0 && y < colorIm.rows-1 && mask.at<unsigned char>(y+1,x) > 0){
						//triA[0].push_back(SpT(i,vindex.at<int>(y-1,x),-lambda_2/4));
						//triA[0].push_back(SpT(i,vindex.at<int>(y+1,x),-lambda_2/4));
						//findByKey(neighbors2[i],vindex.at<int>(y-1,x))->second += -lambda_2/4;
						//findByKey(neighbors2[vindex.at<int>(y+1,x))->second += -lambda_2/4;
						members.push_back(vindex.at<int>(y-1,x));
						members.push_back(vindex.at<int>(y+1,x));
						for (int j=0;j<members.size();j++) {
							findByKey(neighbors2,i,members[j])->second += -lambda_2/4;
							findByKey(neighbors2,members[j],i)->second += -2*lambda_2/16;
						}
						for (int j=0;j<members.size();j++)
							for (int k=0;k<members.size();k++)
								findByKey(neighbors2,members[j],members[k])->second += lambda_2/16;
					}
				//if (x >= colorIm.cols-1 || mask.at<unsigned char>(y,x+1) <= 0){
				if (1) {
					//triA[0].push_back(SpT(i,i,lambda_1 + dcount*lambda_2/4));
					neighbors2[i].begin()->second += lambda_1 + ((double)members.size())*lambda_2/4;
					b_depth(i) = lambda_1*dp;
					continue;
				}
				else {
					//triA[0].push_back(SpT(i,i,lambda_1 + lambda_2 + dcount*lambda_2/4));
					//triA[0].push_back(SpT(i,vindex.at<int>(y,x+1),-lambda_2));
					neighbors2[i].begin()->second += lambda_1 + lambda_2 + ((double)members.size())*lambda_2/4;
					findByKey(neighbors2,i,vindex.at<int>(y,x+1))->second += -lambda_2;
					findByKey(neighbors2,vindex.at<int>(y,x+1),i)->second += -lambda_2;
					neighbors2[vindex.at<int>(y,x+1)].begin()->second += lambda_2;
					b_depth(i) = lambda_1*dp + lambda_2*(dp - refDepth.at<float>(y,x+1));
					continue;
				}
			}

			float d1 = refDepth.at<float>(y,x-1);
			//pp[0] = _k[0]/(x*dp - (x-1)*d1); 
			pp[0] = 1;
			n(0) = pp[0]*(dp-d1);
			//dy
			if (y <= 0 || mask.at<unsigned char>(y-1,x) <= 0 || y >= colorIm.rows-1 || mask.at<unsigned char>(y+1,x) <= 0) {
				std::vector<int> members;
					if (x > 0 && mask.at<unsigned char>(y,x-1) > 0 && x < colorIm.cols-1 && mask.at<unsigned char>(y,x+1) > 0){
						//dcount = 2;
						//triA[0].push_back(SpT(i,vindex.at<int>(y,x-1),-lambda_2/4));
						//triA[0].push_back(SpT(i,vindex.at<int>(y,x+1),-lambda_2/4));
						members.push_back(vindex.at<int>(y,x-1));
						members.push_back(vindex.at<int>(y,x+1));
						for (int j=0;j<members.size();j++) {
							findByKey(neighbors2,i,members[j])->second += -lambda_2/4;
							findByKey(neighbors2,members[j],i)->second += -2*lambda_2/16;
						}
						for (int j=0;j<members.size();j++)
							for (int k=0;k<members.size();k++)
								findByKey(neighbors2,members[j],members[k])->second += lambda_2/16;
					}
				//if (y >= colorIm.rows-1 || mask.at<unsigned char>(y+1,x) <= 0){
				if (1) {
					//triA[0].push_back(SpT(i,i,lambda_1 + dcount*lambda_2/4));
					neighbors2[i].begin()->second += lambda_1 + ((double)members.size())*lambda_2/4;
					b_depth(i) = lambda_1*dp;
					continue;
				}
				else {
					//triA[0].push_back(SpT(i,i,lambda_1 + lambda_2  + dcount*lambda_2/4));
					neighbors2[i].begin()->second += lambda_1 + lambda_2 + ((double)members.size())*lambda_2/4;
					//triA[0].push_back(SpT(i,vindex.at<int>(y+1,x),-lambda_2));
					findByKey(neighbors2,i,vindex.at<int>(y+1,x))->second += -lambda_2;
					findByKey(neighbors2,vindex.at<int>(y+1,x),i)->second += -lambda_2;
					neighbors2[vindex.at<int>(y+1,x)].begin()->second += lambda_2;
					b_depth(i) = lambda_1*dp + lambda_2*(dp - refDepth.at<float>(y+1,x));
					continue;
				}
			}
			float d2 = refDepth.at<float>(y-1,x);
			//pp[1] = _k[4]/(y*dp - (y-1)*d2);
			pp[1] = 1;
			n(1) = pp[1]*(dp-d2);
			//dz
			n(2) = -1;
			double nn = sqrt(n(0)*n(0)+n(1)*n(1)+n(2)*n(2));
			//printf("n %f %f, %f %f %f, %f\n",pp[0],pp[1],n(0),n(1),n(2),nn);
			double w1[3], w2[3];
			for (int k=0;k<3;k++){
				w1[k] = albedo[k](i)*shp[k].at<float>(0,0)*pp[0]/nn;
				w2[k] = albedo[k](i)*shp[k].at<float>(1,0)*pp[1]/nn;
				wb += (w1[k] + w2[k])*(((double)c(k) - x_beta[k](i)) - albedo[k](i)*shp[k].at<float>(2,0)/nn - albedo[k](i)*shp[k].at<float>(3,0));
				wA += (w1[k] + w2[k])*(w1[k] + w2[k]);
				//tmpW[k] = 0;
				//image term
				int ind1 = vindex.at<int>(y,x-1);
				int ind2 = vindex.at<int>(y-1,x);
				findByKey(neighbors2,i,ind1)->second += -(w1[k] + w2[k])*w1[k];
				findByKey(neighbors2,i,ind2)->second += -(w1[k] + w2[k])*w2[k];
				findByKey(neighbors2,ind1,i)->second += -(w1[k] + w2[k])*w1[k];
				findByKey(neighbors2,ind2,i)->second += -(w1[k] + w2[k])*w2[k];
				neighbors2[ind1].begin()->second += w1[k]*w1[k];
				neighbors2[ind2].begin()->second += w2[k]*w2[k];
				findByKey(neighbors2,ind1,ind2)->second += w1[k]*w2[k];
				findByKey(neighbors2,ind2,ind1)->second += w1[k]*w2[k];
			}
			//printf("w %f %f %f, %f %f\n",tmpW[0],tmpW[1],tmpW[2],wb,wA);
			vector<int> members;members.clear();
			if (x < colorIm.cols -1 && mask.at<unsigned char>(y,x+1) > 0) {
				//triA[0].push_back(SpT(i,vindex.at<int>(y,x-1),-tmpW[0]*albedo[0](i)*shp[0].at<float>(0,0)*pp[0]/nn-tmpW[1]*albedo[1](i)*shp[1].at<float>(0,0)*pp[0]/nn
				//	-tmpW[2]*albedo[2](i)*shp[2].at<float>(0,0)*pp[0]/nn - lambda_2/4));  // x-1
				//triA[0].push_back(SpT(i,vindex.at<int>(y,x+1),- lambda_2/4));  // x+1
				members.push_back(vindex.at<int>(y,x-1));
				members.push_back(vindex.at<int>(y,x+1));
			}
			else {
				//triA[0].push_back(SpT(i,vindex.at<int>(y,x-1),-tmpW[0]*albedo[0](i)*shp[0].at<float>(0,0)*pp[0]/nn-tmpW[1]*albedo[1](i)*shp[1].at<float>(0,0)*pp[0]/nn
				//	-tmpW[2]*albedo[2](i)*shp[2].at<float>(0,0)*pp[0]/nn));  // x-1
				//countLap -= 2;
			}
			if (y < colorIm.rows -1  && mask.at<unsigned char>(y+1,x) > 0 ) {
			//triA[0].push_back(SpT(i,vindex.at<int>(y-1,x),-tmpW[0]*albedo[0](i)*shp[0].at<float>(1,0)*pp[1]/nn-tmpW[1]*albedo[1](i)*shp[1].at<float>(1,0)*pp[1]/nn
			//	-tmpW[2]*albedo[2](i)*shp[2].at<float>(1,0)*pp[1]/nn - lambda_2/4));  // y-1
			//triA[0].push_back(SpT(i,vindex.at<int>(y+1,x),- lambda_2/4));  // y+1
				members.push_back(vindex.at<int>(y-1,x));
				members.push_back(vindex.at<int>(y+1,x));
			}
			else {
				//triA[0].push_back(SpT(i,vindex.at<int>(y-1,x),-tmpW[0]*albedo[0](i)*shp[0].at<float>(1,0)*pp[1]/nn-tmpW[1]*albedo[1](i)*shp[1].at<float>(1,0)*pp[1]/nn
				//-tmpW[2]*albedo[2](i)*shp[2].at<float>(1,0)*pp[1]/nn));  // y-1
				//countLap -= 2;
			}
			
			neighbors2[i].begin()->second += wA +lambda_1+lambda_2*(double)(members.size())/4;
			//triA[0].push_back(SpT(i,i,wA+lambda_1+lambda_2*countLap/4));
						for (int j=0;j<members.size();j++) {
							findByKey(neighbors2,i,members[j])->second += -lambda_2/4;
							findByKey(neighbors2,members[j],i)->second += -(double)(members.size())*lambda_2/16;
						}
						for (int j=0;j<members.size();j++)
							for (int k=0;k<members.size();k++)
								findByKey(neighbors2,members[j],members[k])->second += lambda_2/16;
			b_depth(i) = wb + lambda_1*dp;
		}
		
		//FILE* flog = fopen("n.txt","w");
		//for (int i=0;i<pointInd[0].size();i++){
		//	fprintf(flog,"%d (%d %d)",i,pointInd[0][i],pointInd[1][i]);
		//	for (IndWeight::iterator it=neighbors2[i].begin();it != neighbors2[i].end();it++){
		//		fprintf(flog," %d (%d %d = %f)",it->first,pointInd[0][it->first],pointInd[1][it->first], it->second);
		//	}
		//	fprintf(flog,"\n");
		//}
		//fclose(flog);

		for (int i=0;i<pointInd[0].size();i++){
			for (IndWeight::iterator it=neighbors2[i].begin();it != neighbors2[i].end();it++){
				if (it->second != 0)
					triA[0].push_back(SpT(i,it->first,it->second));
			}
		}
		SpMat A_depth(pointInd[1].size(),pointInd[1].size());
		A_depth.setFromTriplets(triA[0].begin(),triA[0].end());
		Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
		//Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
		solver.compute(A_depth);
		//SpMat AT(A_depth.transpose());
		//solver.compute(AT*A_depth);
		if(solver.info()!=Eigen::Success) {
		  // decomposition failed
		  printf("Solving failed!\n");
		}
		x_depth = solver.solve(b_depth);
		//x_depth = solver.solve(AT*b_depth);

		for (int j=0;j<pointInd[0].size();j++){
			int x = pointInd[1][j];
			int y = pointInd[0][j];
			if (x <= 0 || mask.at<unsigned char>(y,x-1) <= 0 /*|| x >= colorIm.cols-1 || mask.at<unsigned char>(y,x+1) <= 0*/) {
				continue;
			}
			if (y <= 0 || mask.at<unsigned char>(y-1,x) <= 0 /*|| y >= colorIm.rows-1 || mask.at<unsigned char>(y+1,x) <= 0*/) {
				continue;
			}

			refDepth.at<float>(pointInd[0][j],pointInd[1][j]) = x_depth(j);
		}
		//refDepth2 = refDepth-minDepth;
		sprintf(text,"newdepth_%d.png",diter);
		imwrite(text,refDepth);
		for (int i=0;i<pointInd[0].size();i++){
			v.at<float>(i,2) = x_depth(i)+minDepth;
			v.at<float>(i,0) = v.at<float>(i,2)*(pointInd[1][i] - _k[2])/_k[0];
			v.at<float>(i,1) = v.at<float>(i,2)*(pointInd[0][i] - _k[5])/_k[4];
		}
		sprintf(text,"%s_sfs_%d.ply",model_file.c_str(),diter);
		write_plyFloat(text,v,cl,fac2);
	}

	//getchar();
	return true;
}


cv::Mat FaceServices2::computeDepthImageNormal(cv::Mat &refDepth,cv::Mat &mask){
	cv::Mat out = Mat::zeros(refDepth.rows,refDepth.cols,CV_32FC3);
	for (int y=0;y<out.rows;y++){
		for (int x=0;x<out.cols;x++){ 
			if (mask.at<unsigned char>(y,x) > 0){
				cv::Vec3f n;
				int u, v;
				// dx
				u = v = x;
				if (x > 0 && mask.at<unsigned char>(y,x-1) > 0) u = x-1;
				if (x < mask.cols-1 && mask.at<unsigned char>(y,x+1) > 0) v = x+1;
				if (u==v) continue;
				float d1 = refDepth.at<float>(y,v);
				float d2 = refDepth.at<float>(y,u);
				n(0) = (d1-d2)/(v-u)  /**_k[0]/(v*d1 - u*d2)*/;
				//dy
				u = v = y;
				if (y > 0 && mask.at<unsigned char>(y-1,x) > 0) u = y-1;
				if (y < mask.rows-1 && mask.at<unsigned char>(y+1,x) > 0) v = y+1;
				if (u==v) continue;
				d1 = refDepth.at<float>(v,x);
				d2 = refDepth.at<float>(u,x);
				n(1) = (d1-d2)/(v-u) /**_k[4]/(v*d1 - u*d2)*/;
				//dz
				n(2) = -1;
				float nn = sqrt(n(0)*n(0)+n(1)*n(1)+n(2)*n(2));
				out.at<Vec3f>(y,x) = n/nn;
			}
		}
	}
	return out;
}

void FaceServices2::computeFaces(cv::Mat findex, vector<cv::Vec3i> &updated_faces){
	int index[3];
	updated_faces.clear();
	for( int r= 0; r< findex.rows - 1; r += 1 ) {
		for( int c= 0; c< findex.cols - 1; c += 1 ) {
			index[0] = findex.at<int>(r,c+1);
			if (index[0] >= 0) {
				index[1] = findex.at<int>(r,c);
				index[2] = findex.at<int>(r+1,c);
				if (index[1] >= 0 && index[2] >= 0)
					updated_faces.push_back(cv::Vec3i(index[0],index[1],index[2]));

				index[1] = findex.at<int>(r+1, c);
				index[2] = findex.at<int>(r+1, c+1);
				if (index[1] >= 0 && index[2] >= 0)
					updated_faces.push_back(cv::Vec3i(index[0],index[1],index[2]));

				if (findex.at<int>(r+1,c) < 0 && findex.at<int>(r,c) >= 0 && findex.at<int>(r+1,c+1) >= 0)
					updated_faces.push_back(cv::Vec3i(index[0],findex.at<int>(r,c),findex.at<int>(r+1,c+1)));
			}
			else
				if (findex.at<int>(r,c+1) >= 0 && findex.at<int>(r+1,c) >= 0 && findex.at<int>(r+1,c+1) >= 0)
					updated_faces.push_back(cv::Vec3i(findex.at<int>(r,c+1),findex.at<int>(r+1,c),findex.at<int>(r+1,c+1)));
		}
	}
}

IndWeight::iterator FaceServices2::findByKey(IndWeight *list, int index,int key){
	for (IndWeight::iterator it = list[index].begin(); it != list[index].end(); it++){
		if (it->first == key) {
			return it;
		}
	}
	printf("Nokey %d %d\n",index,key); getchar();
	return list[index].end();
}
