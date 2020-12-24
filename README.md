IRIS-3DMM Version 1.0 revision 1

An implementation of 3DMM fitting algorithm proposed by [2] and [3] to solve the **3D face modeling from a single image** problem. A modified version was used to generate training data for [1].

## Compilation:
Platforms: Linux

**Important note:** To use the code as is, you will need to ask for an access to the [Basel Face Model database](http://faces.cs.unibas.ch/bfm/main.php?nav=1-1-0&id=details). Without this license, I'm not allowed to redistribute it. 

Send an email to **anstar1111@gmail.com** or **jongmoochoi@gmail.com** with a proof of your permission to use Basel Face Model.  We will give you the binary data (`BaselFace.dat`). Copy it to `lib` directory.

There are 2 options below to compile our code:

### Installation with Docker (recommended)

- Install [Docker CE](https://docs.docker.com/install/)
- With Linux, [manage Docker as non-root user](https://docs.docker.com/install/linux/linux-postinstall/)
- Build docker image:
```
	docker build -t 3dmm-basic .
```
### Installation without Docker on Linux

The steps below have been tested on Ubuntu Linux only:

1. Get Ubuntu 64 bit 12.04 or later 

2. Install **cmake**: 
```
		apt-get install cmake
```
3. Install **[opencv](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html)** (2.4.6 or higher is recommended)
4. Install **boost** library (1.5 or higher is recommended):
```
		apt-get install libboost-all-dev
```
5. Install **OpenGL**, **freeglut**, and **glew**
```
		sudo apt-get install freeglut3-dev
		sudo apt-get install libglew-dev
```
6. Make build directory (temporary). Make & install to bin folder:
```
		mkdir build
		cd build
		cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../bin ..
		make
		make install
```
		
The executable file `IRIS_3DMM` is in folder `bin`

## Start docker container
If you compile our code with Docker, you need to start a Docker container to run our code. You also need to set up a shared folder to transfer input/output data between the host computer and the container.
- Prepare the shared folder on the host computer. For example, `/home/ubuntu/shared`
- Copy input data (if needed) to the shared folder
- Start container:
```
	docker run --rm -ti --ipc=host --privileged -v /home/ubuntu/shared:/shared 3dmm-basic bash
```
Now folder `/home/ubuntu/shared` on your host computer will be mounted to folder `/shared` inside the container. Before exiting the docker container, remember to save your output data to the shared folder.

## Examples:
Model 3D face for a single image (`./script/iris3d_file.sh`) or all images in a folder (`./script/iris3d_dir.sh`)

0. Create output folder (e.g. `./output`)
1. Copy the script into `./bin`
2. Provide the input/output path inside the script
3. Run the script

## Usage
./IRIS_3DMM {paramName paramValue}

Behaviours:
- Output a set of files: 3D model (`.ply`), cropped image (`.ply_cropped.png`), shape parameters (`.alpha`), texture parameters (`.beta`), and render parameters (`.rend`). The render parameters include: rotation angles (3), translation (3), ambient light (3), diffuse light (3), light direction (2), and color model parameters (7).
- Default instrinsic camera matrix: `[-1000 0 w/2; 0 1000 h/2; 0 0 1]` (`w` & `h` are width & height of the cropped image)
- To avoid some runtime errors, this program works on maximum 36 images per run. It saves index of the last processed image into file `startF.txt`. You can continue to process the next images by passing parameter `"-con 1"` in the next run. After finishing, it saves `-1` into `startF.txt`.
- If the output model file (`.ply`) exists, the program will skip it and goes to the next one. Hence, if you want to re-generates 3D models, you need to clean up the output folder before running.

Parameters:
- General:
```
	-mloc    : path of ccnf model
	-fdloc   : path of face detector cascade
	-clmwild : (optinal) use trained model for images in the wild. No value is needed
	-con	 : (optinal) model images from the begining (0-default), or from a specifice index defined in file startF.txt
		   E.g. when the input folder has 300 images, but you want to model the last 100 images only, you can set the number 
in startF as 200 and run the program with parameter "-con 1"
```
	   
- From file
```
	-f	 : input file
	-of	 : output file
```
- From directory
```
	-fdir	 : input directory
	-ofdir	 : output directory
```

## References

[1] A. Tran, T. Hassner, I. Masi, G. Medioni, "[Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network](https://arxiv.org/abs/1612.04904)", arxiv pre-print 2016 

[2] V. Blanz, T. Vetter, "[Face recognition based on fitting a 3D morphable model](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1227983)", IEEE Transactions on pattern analysis and machine intelligence 25, no. 9 (2003): 1063-1074.

[3] S. Romdhani, "[Face image analysis using a multiple features fitting strategy](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.471.3366&rep=rep1&type=pdf)", PhD diss., University_of_Basel, 2005.

[4] T. Baltrusaitis, P. Robinson, L. P. Morency, "[Constrained local neural fields for robust facial landmark detection in the wild](https://www.cl.cam.ac.uk/~tb346/pub/papers/iccv2013.pdf)", In Proceedings of the IEEE International Conference on Computer Vision Workshops, pp. 354-361. 2013.

## Changelog
- Dec 2017, First Release 

## Disclaimer

_The SOFTWARE PACKAGE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use._

## Contacts

If you have any questions, drop an email to _anhttran@usc.edu_ or leave a message below with GitHub (log-in is needed).



