/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
//#include "cv.h"
//#include "highgui.h"
#include "FImRenderer.h"
#include "BaselFaceEstimator.h"
#include "RenderModel.h"
#include <Eigen/Sparse>

using namespace std;
using namespace cv;

#define NUM_EXTRA_FEATURES 3
#define FEATURES_LANDMARK	  0
#define FEATURES_TEXTURE_EDGE 1
#define FEATURES_CONTOUR_EDGE 2
#define FEATURES_TEXTURE_CNST 3
#define FEATURES_SPECULAR	  4

#define TEXEDGE_ORIENTATION_REG_NUM 8
#define NUM_AREA_BIN 1000

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> SpT;
typedef std::vector<std::pair<int, double> > IndWeight;

typedef struct BFMParams {
	float sI;
	float sS[6];
	float sF[NUM_EXTRA_FEATURES];
	float sR[RENDER_PARAMS_COUNT];
	float initR[RENDER_PARAMS_COUNT];
	bool  doOptimize[RENDER_PARAMS_COUNT];
	bool  optimizeAB[2];
	bool  optimizeExpr;
	bool  computeEI;
	float sExpr;
	
	int indexCSArea[NUM_AREA_BIN];
	cv::Mat triVis;
	cv::Mat triNoShadow;
	cv::Mat triAreas;
	cv::Mat triCSAreas;
	cv::Mat hessDiag;  
	cv::Mat gradVec;  

	std::vector<int> texEdgeVisIndices;
	std::vector<int> texEdgeVisBin;
	std::vector<int> conEdgeIndices;
	std::vector<int> conEdgeBin;

	void init(){
		// init
		sS[0] = sS[2] = 0;
		sS[1] = 0;
		sS[3] = sS[4] = sS[5] = 0;
		memset(doOptimize,false,sizeof(bool)*RENDER_PARAMS_COUNT);
		optimizeAB[0] = true;
		optimizeAB[1] = false;
		optimizeExpr = true;
		computeEI = false;
		sExpr = 1;
		for (int i=0;i<6;i++) doOptimize[i] = true;

		if (RENDER_PARAMS_AMBIENT < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_AMBIENT+i] = RENDER_PARAMS_AMBIENT_DEFAULT;
		}
		if (RENDER_PARAMS_DIFFUSE < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_DIFFUSE+i] = 0.0f;
		}
		if (RENDER_PARAMS_LDIR < RENDER_PARAMS_COUNT){
			for (int i=0;i<2;i++) initR[RENDER_PARAMS_LDIR+i] = 0.0f;
		}
		if (RENDER_PARAMS_CONTRAST < RENDER_PARAMS_COUNT){
			initR[RENDER_PARAMS_CONTRAST] = RENDER_PARAMS_CONTRAST_DEFAULT;
		}
		if (RENDER_PARAMS_GAIN < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_GAIN+i] = RENDER_PARAMS_GAIN_DEFAULT;
		}
		if (RENDER_PARAMS_OFFSET < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_OFFSET+i] = RENDER_PARAMS_OFFSET_DEFAULT;
		}
		if (RENDER_PARAMS_SPECULAR < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_SPECULAR+i] = RENDER_PARAMS_SPECULAR_DEFAULT;
		}
		if (RENDER_PARAMS_SHINENESS < RENDER_PARAMS_COUNT){
			initR[RENDER_PARAMS_SHINENESS] = RENDER_PARAMS_SHINENESS_DEFAULT;
		}
		
		// sR
		for (int i=0;i<3;i++) sR[RENDER_PARAMS_R+i] = (M_PI/6)*(M_PI/6);
		for (int i=0;i<3;i++) sR[RENDER_PARAMS_T+i] = 900.0f;
		if (RENDER_PARAMS_AMBIENT < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_AMBIENT+i] = 1;
		}
		if (RENDER_PARAMS_DIFFUSE < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_DIFFUSE+i] = 1;
		}
		if (RENDER_PARAMS_LDIR < RENDER_PARAMS_COUNT){
			for (int i=0;i<2;i++) sR[RENDER_PARAMS_LDIR+i] = M_PI*M_PI;
		}
		if (RENDER_PARAMS_CONTRAST < RENDER_PARAMS_COUNT){
			sR[RENDER_PARAMS_CONTRAST] = 1;
		}
		if (RENDER_PARAMS_GAIN < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_GAIN+i] = 4.0f;
		}
		if (RENDER_PARAMS_OFFSET < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_OFFSET+i] = 10000.0f;
		}
		if (RENDER_PARAMS_SPECULAR < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_SPECULAR+i] = 10000.0f;
		}
		if (RENDER_PARAMS_SHINENESS < RENDER_PARAMS_COUNT){
			sR[RENDER_PARAMS_SHINENESS] = 1000000.0f;
		}
	}
} BFMParams;

typedef struct BFMSymParams {
	float sI;
	float sS[6];
	float sF[NUM_EXTRA_FEATURES];
	float sR[RENDER_PARAMS_COUNT];
	float initR[RENDER_PARAMS_COUNT];
	bool  doOptimize[RENDER_PARAMS_COUNT];
	bool  optimizeAB[2];
	bool  optimizeExpr;
	bool  computeEI;
	float sExpr;
	cv::Mat gradVec;  
	
	int indexCSArea[NUM_AREA_BIN];
	cv::Mat triVis[2];
	cv::Mat triNoShadow[2];
	cv::Mat triAreas[2];
	cv::Mat trisumAreas;
	cv::Mat triCSAreas;
	cv::Mat hessDiag;  

	std::vector<int> texEdgeVisIndices[2];
	std::vector<int> texEdgeVisBin[2];
	std::vector<int> conEdgeIndices[2];
	std::vector<int> conEdgeBin[2];

	void init(){
		// init
		sS[0] = sS[2] = 0;
		sS[1] = 0;
		sS[3] = sS[4] = sS[5] = 0;
		memset(doOptimize,false,sizeof(bool)*RENDER_PARAMS_COUNT);
		optimizeAB[0] = true;
		optimizeAB[1] = false;
		optimizeExpr = true;
		computeEI = false;
		sExpr = 1;
		for (int i=0;i<6;i++) doOptimize[i] = true;

		if (RENDER_PARAMS_AMBIENT < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_AMBIENT+i] = RENDER_PARAMS_AMBIENT_DEFAULT;
		}
		if (RENDER_PARAMS_DIFFUSE < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_DIFFUSE+i] = 0.0f;
		}
		if (RENDER_PARAMS_LDIR < RENDER_PARAMS_COUNT){
			for (int i=0;i<2;i++) initR[RENDER_PARAMS_LDIR+i] = 0.0f;
		}
		if (RENDER_PARAMS_CONTRAST < RENDER_PARAMS_COUNT){
			initR[RENDER_PARAMS_CONTRAST] = RENDER_PARAMS_CONTRAST_DEFAULT;
		}
		if (RENDER_PARAMS_GAIN < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_GAIN+i] = RENDER_PARAMS_GAIN_DEFAULT;
		}
		if (RENDER_PARAMS_OFFSET < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_OFFSET+i] = RENDER_PARAMS_OFFSET_DEFAULT;
		}
		if (RENDER_PARAMS_SPECULAR < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_SPECULAR+i] = RENDER_PARAMS_SPECULAR_DEFAULT;
		}
		if (RENDER_PARAMS_SHINENESS < RENDER_PARAMS_COUNT){
			initR[RENDER_PARAMS_SHINENESS] = RENDER_PARAMS_SHINENESS_DEFAULT;
		}
		
		// sR
		for (int i=0;i<3;i++) sR[RENDER_PARAMS_R+i] = (M_PI/6)*(M_PI/6);
		for (int i=0;i<3;i++) sR[RENDER_PARAMS_T+i] = 900.0f;
		if (RENDER_PARAMS_AMBIENT < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_AMBIENT+i] = 1;
		}
		if (RENDER_PARAMS_DIFFUSE < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_DIFFUSE+i] = 1;
		}
		if (RENDER_PARAMS_LDIR < RENDER_PARAMS_COUNT){
			for (int i=0;i<2;i++) sR[RENDER_PARAMS_LDIR+i] = M_PI*M_PI;
		}
		if (RENDER_PARAMS_CONTRAST < RENDER_PARAMS_COUNT){
			sR[RENDER_PARAMS_CONTRAST] = 1;
		}
		if (RENDER_PARAMS_GAIN < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_GAIN+i] = 4.0f;
		}
		if (RENDER_PARAMS_OFFSET < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_OFFSET+i] = 10000.0f;
		}
		if (RENDER_PARAMS_SPECULAR < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_SPECULAR+i] = 10000.0f;
		}
		if (RENDER_PARAMS_SHINENESS < RENDER_PARAMS_COUNT){
			sR[RENDER_PARAMS_SHINENESS] = 1000000.0f;
		}
	}
} BFMSymParams;

class FaceServices2
{
	float _k[9];
	FImRenderer* im_render;
	FImRenderer* im_render2;
	BaselFaceEstimator festimator;
	RenderServices rs;
	
	float prevEI;
	float prevEF;
	float cEF;
	float cEI;
	float cETE;
	float cECE;
	float cES;
	float mstep;
	int countFail;
	float maxVal;
	float mlambda;

	float TexAngleCenter[TEXEDGE_ORIENTATION_REG_NUM];
	cv::Mat edgeAngles;
	cv::Mat *texEdge, *conEdge;
	cv::Mat *texEdgeDist, *conEdgeDist;
	cv::Mat *texEdgeDistDX, *texEdgeDistDY, *conEdgeDistDX, *conEdgeDistDY;
	std::vector<int> texEdgeIndices;
public:
	FaceServices2(void);
	void setUp(int w, int h, float f);
	static cv::Mat *symSPC;
	static cv::Mat *symTPC;
	bool projectCheckVis(FImRenderer* imRen, cv::Mat shape, float* r, float *t, cv::Mat refDepth, bool* &visible);
	std::vector<cv::Point2f> projectCheckVis2(FImRenderer* imRen, cv::Mat shape, float* r, float *t, cv::Mat refDepth, bool* &visible);
	bool singleFrameRecon(cv::Mat colorIm, cv::Mat lms,cv::Vec6d poseCLM, float conf,cv::Mat lmVis, cv::Mat &shape, cv::Mat &tex,string model_file = "", string lm_file = "", string pose_file = "", string refDir = "");
	bool updateTriangles(cv::Mat colorIm,cv::Mat faces,bool part, cv::Mat alpha, float* renderParams, BFMParams &params, cv::Mat exprW = cv::Mat() );
	
	float updateHessianMatrix(bool part, cv::Mat alpha, cv::Mat beta, float* renderParams, cv::Mat faces, cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat exprW = cv::Mat(), bool show = false );
	cv::Mat computeGradient(bool part, cv::Mat alpha, cv::Mat beta, float* renderParams, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params,std::vector<int> &inds, cv::Mat exprW);
	void sno_step(bool part, cv::Mat &alpha, cv::Mat &beta, float* renderParams, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW);
	void sno_step2(bool part, cv::Mat &alpha, cv::Mat &beta, float* renderParams, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW);
	float line_search(bool part, cv::Mat &alpha, cv::Mat &beta, float* renderParams, cv::Mat &dirMove,std::vector<int> inds, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW, int maxIters = 4);
	float computeCost(float vEF, float vEI, float vETE, float vECE, float cS, cv::Mat &alpha, cv::Mat &beta, float* renderParams, BFMParams &params, cv::Mat &exprW );
	
	cv::Vec6d vS(cv::Mat &alpha, cv::Mat &beta);
	float eS(cv::Mat &alpha, cv::Mat &beta, BFMParams &Params);
	float eE(cv::Mat colorIm, std::vector<cv::Point2f> projPoints, BFMParams &Params, int type);
	float eF(bool part, cv::Mat alpha, std::vector<int> inds, cv::Mat landIm, float* renderParams, cv::Mat exprW);
	float eI(cv::Mat colorIm,cv::Mat &centerPoints, cv::Mat &centerTexs, cv::Mat &normals, float* renderParams,std::vector<int> inds, BFMParams &params, bool show = false);
	void getTrianglesCenterNormal(bool part, cv::Mat alpha,cv::Mat beta, cv::Mat faces,std::vector<int> inds0, cv::Mat &centerPoints, cv::Mat &centerTexs, cv::Mat &normals, cv::Mat exprW);
	void getTrianglesCenterVNormal(bool part, cv::Mat alpha, cv::Mat faces,std::vector<int> inds0, cv::Mat &centerPoints, cv::Mat &normals, cv::Mat exprW);
	void getTrianglesCenterTex(bool part, cv::Mat beta, cv::Mat faces,std::vector<int> inds0, cv::Mat &centerTexs);
	std::vector<int> verticesFromTriangles(cv::Mat faces,std::vector<int> inds0);
	std::vector<int> aVerticesFromTriangles(cv::Mat faces,std::vector<int> inds0, int vind = 0);
	void randSelectTriangles(int numPoints, BFMParams &params, std::vector<int> &inds);
	
	void write_SelectedTri(char* fname, cv::Mat shape, cv::Mat faces,std::vector<int> inds, cv::Mat centerPoints, cv::Mat centerTexs, cv::Mat normals);
	void write_FaceArea(char* fname, cv::Mat shape, cv::Mat faces, cv::Mat &vis, cv::Mat &areas);
	void write_FaceShadow(char* fname, cv::Mat shape, cv::Mat faces, cv::Mat &vis, cv::Mat &noShadow);
	void renderFace(char* fname, cv::Mat colorIm, cv::Mat landIm,bool part, cv::Mat alpha, cv::Mat beta,cv::Mat faces, float* renderParams, cv::Mat exprW );
	bool loadReference(string refDir, string model_file, cv::Mat &alpha, cv::Mat &beta, float* renderParams, int &M, cv::Mat &exprW, int &EM);
	
	bool prepareEdgeDistanceMaps(cv::Mat colorIm);
	void initRenderer(cv::Mat &colorIm);
	void mergeIm(cv::Mat* output,cv::Mat bg,cv::Mat depth);
	~FaceServices2(void);

	
	void randSelectTrianglesSym(int numPoints, BFMSymParams &params, std::vector<int> &inds);
	bool singleFrameReconSym(cv::Mat colorIm, cv::Mat lms,cv::Vec6d poseCLM,float conf,cv::Mat lmVis,cv::Mat &shape,cv::Mat &tex,string model_file = "", string lm_file = "", string pose_file = "", string refDir = "");
	
	cv::Mat computeGradientSym(bool part, cv::Mat &alpha, cv::Mat &beta,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat faces,std::vector<cv::Mat> colorIms,std::vector<std::vector<int> > lmInds, std::vector<cv::Mat> landIms, BFMSymParams &params, std::vector<int> &inds, cv::Mat exprW);
	void sno_step2Sym(bool part, cv::Mat &alpha, cv::Mat &beta,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat faces,std::vector<cv::Mat> colorIms,std::vector<std::vector<int> > lmInds, std::vector<cv::Mat> landIms, BFMSymParams &params, cv::Mat &exprW);
	float updateHessianMatrixSym(bool part, cv::Mat &alpha, cv::Mat &beta,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat faces, std::vector<cv::Mat> colorIms,std::vector<std::vector<int> > lmInds, std::vector<cv::Mat> landIms, BFMSymParams &params, cv::Mat &exprW, bool show = false );
	void renderFaceSym(char* fname, cv::Mat colorIm, bool part, cv::Mat alpha, cv::Mat beta,cv::Mat faces,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat exprW );
	bool updateTrianglesSym(cv::Mat colorIm,cv::Mat faces,bool part, cv::Mat alpha,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, BFMSymParams &params, cv::Mat exprW );
	float eISym(std::vector<cv::Mat> &ims,cv::Mat* centerPointsArr, cv::Mat centerTexs, cv::Mat* normalsArr,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams,std::vector<int> inds, BFMSymParams &paramss, cv::Mat &exprW, bool show = false);
	float eFSym(bool part, cv::Mat alpha, std::vector<std::vector<int> > inds, std::vector<cv::Mat> landIms,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat &exprW, bool show = false);
	float line_searchSym(bool part, cv::Mat &alpha, cv::Mat &beta,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, cv::Mat &dirMove,std::vector<int> inds, cv::Mat faces, std::vector<cv::Mat> colorIms,std::vector<std::vector<int> > lmInds, std::vector<cv::Mat> landIms, BFMSymParams &params, cv::Mat &exprW, int maxIters = 4);
	float computeCostSym(float vEF, float vEI, float vETE, float vECE,float vES, cv::Mat &alpha, cv::Mat &beta,std::vector<cv::Mat> &r3Ds,std::vector<cv::Mat> &t3Ds, float* renderParams, BFMSymParams &params, cv::Mat &exprW );
	float eESym(cv::Mat colorIm, std::vector<cv::Point2f> projPoints[], BFMSymParams &Params, int type);
	float eS(cv::Mat &alpha, cv::Mat &beta, BFMSymParams &Params);
	void getTrianglesCenterNormalSym(bool part, cv::Mat alpha,cv::Mat beta, cv::Mat faces,std::vector<int> inds0, cv::Mat* centerPointsArr, cv::Mat &centerTexs, cv::Mat* normalsArr, cv::Mat exprW);
	void getTrianglesCenterVNormalSym(bool part, cv::Mat alpha, cv::Mat faces,std::vector<int> inds0, cv::Mat* centerPointsArr, cv::Mat* normalsArr, cv::Mat exprW);

	bool sfs(cv::Mat colorIm, cv::Mat lms,cv::Vec6d poseCLM, float conf,cv::Mat lmVis, cv::Mat &shape, cv::Mat &tex,string model_file = "", string lm_file = "", string pose_file = "", string refDir = "");
	cv::Mat computeDepthImageNormal(cv::Mat &refDepth,cv::Mat &mask);
	void computeFaces(cv::Mat vindex, vector<cv::Vec3i> &updated_faces);
	IndWeight::iterator findByKey(IndWeight *list, int index,int key);

	bool loadReference2(string refDir, string model_file, cv::Mat &alpha, cv::Mat &beta, int &M, bool bySID=true);
	bool rendWFixedShape(cv::Mat colorIm, cv::Mat lms,cv::Vec6d poseCLM, float conf,cv::Mat lmVis, cv::Mat &shape, cv::Mat &tex, string model_file, string lm_file, string pose_file, string refDir);
};


