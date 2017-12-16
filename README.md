IRIS-3DMM Version 1.0 revision 1

if further information needed contact:

	Anh Tran (anhttran@usc.edu)

## Compilation:
Platforms: Linux

	0. Get Ubuntu 64 bit 12.04 or later 
	1. Download [BaselFace.dat](https://drive.google.com/file/d/1N_d95ZUDSHk5RHD4X0-TnNj5pZGJqW55/view?usp=sharing) and copy it to `lib` directory.
	2. Install cmake: 
		apt-get install cmake
	3. Install opencv (2.4.6 or higher is recommended):
		(http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html)
	4. Install boost library (1.5 or higher is recommended):
		apt-get install libboost-all-dev
	5. Install OpenGL, freeglut, and glew
		sudo apt-get install freeglut3-dev
		sudo apt-get install libglew-dev
	6. Make build directory (temporary). Make & install to bin folder
		mkdir build
		cd build
		cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../bin ..
		make
		make install
		
The executable file (IRIS_3DMM) is in folder ./bin

## Run IRIS-3DMM on Janus dataset:
	0. Create output folder (e.g. ./output/DS1)
	1. Copy the script ./script/iris3d_meta.sh into ./bin
	2. Provide the directories you want to use inside the script:
		- Root path of CS0/CS2 dataset: DATASET_ROOT_PATH='<your path>'
		- Meta-data file: DATASET_META_FILE='<your path>'
		- Output folder: OUTPUT_DIR='<your path>'
		- Input landmark folder (optional): INPUT_LANDMARK_CMD='-flm <your path', or keep it blank (''). If INPUT_LANDMARK_CMD=='', the program will detect landmarks itself.
	3. Run the script

## Other examples:
	Model 3D face for a single image (./script/iris3d_file.sh) or all images in a folder (./script/iris3d_dir.sh)
	0. Create output folder (e.g. ./output)
	1. Copy the script into ./bin
	2. Provide the input/output path inside the script
	3. Run the script

## Usage
./IRIS_3DMM {<param name> <param value>}

Behavious:
- Output a set of files: 3D model (.ply), cropped image (.ply_cropped.png), shape parameters (.alpha), texture parameters (.beta), and render parameters (.rend). The render parameters include: rotation angles (3), translation (3), ambient light (3), diffuse light (3), light direction (2), color model parameters (7).
- Default instrinsic camera matrix: [-1000 0 w/2; 0 1000 h/2; 0 0 1] (w & h are width & height of the cropped image)
- To avoid some runtime error, this program works on maximum 36 images per run. It saves index of the lasted processed image into file startF.txt. You can continue to process the next images by passing parameter "-con 1" in the next run. After finishing, it saves -1 into startF.txt
- If the output model file (.ply) exists, the program will skip it and goes to the next one. Hence, if you want to re-generates 3D models, you need to clean up the output folder before running.
- With Janus dataset, the output filename will be in this format: <media id>[_<frame id>]_S<subject id>
  While CLNF uses template id in the output filename, I skip it since an image can belong to multiple templates. Thus, my program can avoid wasting time & space to model a face in an image multiple times.

Parameters:
*General:
	-mloc    : path of ccnf model
	-fdloc   : path of face detector cascade
	-clmwild : (optinal) use trained model for images in the wild. No value is needed
	-con	 : (optinal) model images from the begining (0-default), or from a specifice index defined in file startF.txt
		   E.g. when the input folder has 300 images, but you want to model the last 100 images only, you can set the number in startF as 200 and run the program with parameter "-con 1"
	   
*From file
	-f	 : input file
	-of	 : output file

*From directory
	-fdir	 : input directory
	-ofdir	 : output directory
	
*From Janus dataset 
	-ipath	 : Janus dataset root path
	-opath	 : output directory
	-fmeta   : path of the meta-data file
	-flm	 : (optinal) pre-detected landmarks directory. This directory should contains files generated from CLNF, with both point files (.pts) and confidence value files (.cfd)
	-eye_type: (optinal) type of input images: 1-eye images (1), 2-eye images (2), or both of them (0-default)





