#!/bin/bash
INPUT_DIR_PATH='../data/images';
OUTPUT_DIR_PATH='../output/images';
mkdir -p $OUTPUT_DIR_PATH
eval "./IRIS_3DMM -fdir $INPUT_DIR_PATH -ofdir $OUTPUT_DIR_PATH -mloc model/main_ccnf_general.txt -fdcloc classifiers/haarcascade_frontalface_alt.xml"

s=0;
if [ -f startF.txt ];
then
  s=`cat startF.txt`
  echo $s
fi

while [ $s -ne -1 ]
do
	"./IRIS_3DMM -fdir $INPUT_DIR_PATH -ofdir $OUTPUT_DIR_PATH -mloc model/main_ccnf_general.txt -fdcloc classifiers/haarcascade_frontalface_alt.xml -con 1"
	if [ -f startF.txt ];
	then
	  s=`cat startF.txt`
          echo $s
	fi
done
