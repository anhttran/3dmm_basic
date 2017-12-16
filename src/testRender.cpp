#include <stdio.h>
#include "cv.h"
#include "highgui.h"
#include "BaselFaceEstimator.h"
#include "FImRenderer.h"
#include "RenderModel.h"
#include "FaceServices2.h"

cv::Mat loadVals(char* fname, int M){
	cv::Mat w(M,1,CV_32F);
	char text[100];
	FILE* pose = fopen(fname,"r");
	text[0] = '\0';
	for (int i=0;i<M;i++){
	fgets(text,1000,pose);
	int l = strlen(text);
	if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
	w.at<float>(i,0) = atof(text);
	}
	fclose(pose);
	//std::cout << "w " << w << std::endl;
	return w;
}

cv::Mat loadWeight(char* fname, int M){
	cv::Mat w(4*M,1,CV_32F);
	char text[1000];
	FILE* pose = fopen(fname,"r");
	text[0] = '\0';
	for (int i=0;i<4*M;i++){
	fgets(text,1000,pose);
	int l = strlen(text);
	if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
	w.at<float>(i,0) = atof(text);
	}
	fclose(pose);
	std::cout << "w " << w << std::endl;
	return w;
}
//
//cv::Mat skew(cv::Mat v1){
//	cv::Mat out(3,3,CV_32F);
//	out.at<float>(0,0) = out.at<float>(1,1) = out.at<float>(2,2) = 0;
//	out.at<float>(0,1) = -v1.at<float>(2,0);
//	out.at<float>(0,2) = v1.at<float>(1,0);
//	out.at<float>(1,0) = v1.at<float>(2,0);
//	out.at<float>(1,2) = -v1.at<float>(0,0);
//	out.at<float>(2,0) = -v1.at<float>(1,0);
//	out.at<float>(2,1) = v1.at<float>(0,0);
//	return out;
//}
//
//void groundScale(cv::Mat input, cv::Mat &output, float bgThresh, float gapPc) {
//	cv::Mat mask = (input < bgThresh) & (input > 1 - bgThresh);
//	double mn, mx;
//	cv::minMaxLoc(input,&mn,&mx,0,0,mask);
//	printf("mn, mx: %f %f\n",mn,mx);
//	double range = mx - mn;
//	mn = mn - gapPc * range;
//	mn = (mn > 0)?mn:0;
//	mx = mx + gapPc * range;
//	mx = (mx < 1)?mx:1;
//	range = mx - mn;
//
//	cv::Mat mask1 = (input >= bgThresh)/255;
//	mask = mask/255;
//	mask.convertTo(mask,input.type());
//	mask1.convertTo(mask1,input.type());
//	cv::Mat mask2 = 1 - mask - mask1;
//	output = mask.mul((input - mn)/range) + mask1 * 1/* + mask2 * mn*/;
//	//cv::imshow("out",output); cv::waitKey();
//	//output = output*255;
//	//output.convertTo(output,CV_8U);
//}
//
//cv::Mat findRotation(cv::Mat v1, cv::Mat v2){
//	cv::Mat ab = v1.cross(v2);
//	std::cout << "cross " << ab << std::endl;
//	float s = sqrt(ab.at<float>(0,0)*ab.at<float>(0,0) + ab.at<float>(1,0)*ab.at<float>(1,0) + ab.at<float>(2,0)*ab.at<float>(2,0));
//	if (s == 0)
//		return cv::Mat::eye(3,3,CV_32F);
//
//	float c = v1.at<float>(0,0)*v2.at<float>(0,0) + v1.at<float>(1,0)*v2.at<float>(1,0) + v1.at<float>(2,0)*v2.at<float>(2,0);
//	cv::Mat sk = skew(ab);
//	return cv::Mat::eye(3,3,CV_32F) + sk + sk*sk*(1-c)/(s*s);
//}
//
//void main(){
//	BaselFaceEstimator festimator;
//	int M = 30;
//	cv::Mat ws = loadWeight("alpha2.txt",M);
//	cv::Mat wt = loadWeight("beta2.txt",M);
//	cv::Mat shape = festimator.getShapeParts(ws);
//	cv::Mat tex = festimator.getTextureParts(wt);
//	cv::Mat faces = festimator.getFaces();
//	cv::Mat colors;
//	std::cout << shape(cv::Rect(0,0,3,5)) << std::endl;
//	write_plyFloat("tmp.ply",shape,tex,faces-1);
//
//	FImRenderer im_render(cv::Mat::zeros(600,800,CV_8UC3));
//	im_render.loadPLYFile("tmp.ply");
//	//im_render.computeNormals();
//
//	float r[3], t[3];
//	cv::Mat rm(3,1,CV_32F,r);
//	cv::Mat tm(3,1,CV_32F,t);
//	for (int i=0;i<3;i++){
//		r[i] = 0.00001;
//		t[i] = 0.;
//	}
//	//r[1] = -3.14/2;
//	r[1] = 0.5f;
//	t[2] = -500;
//	cv::Mat refRGB = cv::Mat::zeros(600,800,CV_8UC3);
//	cv::Mat refDepth = cv::Mat::zeros(600,800,CV_32F);
//
//	float render_model[RENDER_PARAMS_COUNT] = {r[0], r[1], r[2], t[0], t[1], t[2], 
//												0.2f, 0.2f, 0.2f, 0.8f, 0.8f, 0.8f, 0.0f,-0.5f,
//												0.0f, 1.0f, 1.0f, 1.0f, 100.0f, 0.0f, 0.0f/*, 255.f, 255.f, 255.f, 32.0f*/};
//	bool* visible = new bool[im_render.face_->mesh_.nVertices_];
//	bool* noShadow = new bool[im_render.face_->mesh_.nVertices_];
//
//	RenderServices rs;
//	FaceServices2 fservices;
//	fservices.setUp(800,600,1000.0f);
//	//rs.estimateVertexNormals(shape, faces-1, colors);
//	//rs.estimateColor(shape,tex,faces-1,visible,noShadow,render_model,colors);
//	//im_render.copyColors(colors);
//	im_render.loadModel();
//	im_render.render(r,t,1000.0f,refRGB,refDepth);
//	fservices.projectCheckVis(&im_render, shape, r, t, refDepth, visible);
//	//imwrite("pc.png",refRGB);
//
//	cv::Mat hgD1;
//	cv::Mat hgD3;
//	groundScale(refDepth,hgD1,0.9999,0.1);
//	cv::Mat outDepth = hgD1 * 255;
//	outDepth.convertTo(outDepth,CV_8U);
//	cv::Mat outDepth3(outDepth.rows,outDepth.cols,CV_8UC3);
//	insertChannel(outDepth,outDepth3,0);
//	insertChannel(outDepth,outDepth3,1);
//	insertChannel(outDepth,outDepth3,2);
//	cv::imwrite("pcd.png", outDepth3);
//
//	cv::Mat trgA(3,1,CV_32F);
//	trgA.at<float>(0,0) = 0.0f;
//	trgA.at<float>(1,0) = 0.0f;
//	trgA.at<float>(2,0) = 1.0f;
//	cv::Mat vecL(3,1,CV_32F);
//	vecL.at<float>(0,0) = cos(render_model[RENDER_PARAMS_LDIR])*sin(render_model[RENDER_PARAMS_LDIR+1]);
//	vecL.at<float>(1,0) = sin(render_model[RENDER_PARAMS_LDIR]);
//	vecL.at<float>(2,0) = cos(render_model[RENDER_PARAMS_LDIR])*cos(render_model[RENDER_PARAMS_LDIR+1]);
//	cv::Mat matR = findRotation(vecL,trgA);
//	cv::Mat matR1;
//	cv::Rodrigues(rm,matR1);
//	cv::Mat matR2;
//	matR2 = matR*matR1;
//
//	float r2[3];
//	cv::Mat vecR2(3,1,CV_32F,r2);
//	cv::Rodrigues(matR2,vecR2);
//
//	cv::Mat refRGB2 = cv::Mat::zeros(600,800,CV_8UC3);
//	cv::Mat refDepth2 = cv::Mat::zeros(600,800,CV_32F);
//	im_render.render(r2,t,1000.0f,refRGB2,refDepth2);
//	fservices.projectCheckVis(&im_render, shape, r2, t, refDepth2, noShadow);
//	
//	groundScale(refDepth2,hgD3,0.9999,0.1);
//	outDepth = hgD3 * 255;
//	outDepth.convertTo(outDepth,CV_8U);
//	outDepth3 = cv::Mat::zeros(outDepth.rows,outDepth.cols,CV_8UC3);
//	insertChannel(outDepth,outDepth3,0);
//	insertChannel(outDepth,outDepth3,1);
//	insertChannel(outDepth,outDepth3,2);
//	cv::imwrite("pcd_l.png", outDepth3);
//
//	
//	rs.estimateColor(shape,tex,faces-1,visible,noShadow,render_model,colors);
//	im_render.copyColors(colors);
//	im_render.loadModel();
//	refRGB = cv::Mat::zeros(600,800,CV_8UC3);
//	refDepth = cv::Mat::zeros(600,800,CV_32F);
//	im_render.render(r,t,1000.0f,refRGB,refDepth);
//	imwrite("pc.png",refRGB);
//	refRGB2 = cv::Mat::zeros(600,800,CV_8UC3);
//	refDepth2 = cv::Mat::zeros(600,800,CV_32F);
//	im_render.render(r2,t,1000.0f,refRGB2,refDepth2);
//	imwrite("pc_l.png",refRGB2);
//	//getchar();
//}


int main(){
	char imPath[200] = "/media/anh/Lacie/Anh/CS0/allA/gal/split1/test_1_A_edge/landmark/00245_S54.ply_cropped.png";
	char alphaPath[200] = "/media/anh/Lacie/Anh/CS0/allA/gal/split1/test_1_A_edge/landmark/00245_S54.ply.alpha";
	char betaPath[200] = "/media/anh/Lacie/Anh/CS0/allA/gal/split1/test_1_A_edge/landmark/00245_S54.ply.beta";
	char rendPath[200] = "/media/anh/Lacie/Anh/CS0/allA/gal/split1/test_1_A_edge/landmark/00245_S54.ply.rend";
	FaceServices2  fservice;
	cv::Mat im = imread(imPath);
	fservice.setUp(im.cols,im.rows,1000);
	fservice.initRenderer(im);
	cv::Mat alpha = loadVals(alphaPath, 99);
	cv::Mat beta = loadVals(betaPath, 99);
	cv::Mat rend = loadVals(rendPath, 99);
	float renderParams[21];
	for (int i=0;i<21;i++)
		renderParams[i] = rend.at<float>(i,0);
	BaselFaceEstimator festimator;
	cv::Mat facefill = festimator.getFaces_fill();
	cv::Mat landIm;
	fservice.renderFace("out.png",im,landIm,false,alpha,beta,facefill,renderParams);
return 0;
}
