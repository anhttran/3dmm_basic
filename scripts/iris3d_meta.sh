#!/bin/bash
DATASET_ROOT_PATH='/media/anh/OS/Anh/CS2/';
DATASET_META_FILE='/media/anh/OS/Anh/CS2/protocol/metadata.csv';
OUTPUT_DIR='/media/anh/EXTRA/CS2_data/3DModeling/WEdge2/'
INPUT_LANDMARK_CMD=''
#INPUT_LANDMARK_CMD='-flm ~/Data/CS0/landmarks/'

eval "./IRIS_3DMM -ipath $DATASET_ROOT_PATH -eye_type 0 -opath $OUTPUT_DIR -mloc model/main_ccnf_wild_70.txt -fdcloc classifiers/haarcascade_frontalface_alt.xml -fmeta $DATASET_META_FILE -clmwild $INPUT_LANDMARK_CMD"

s=0;
if [ -f startF.txt ];
then
  s=`cat startF.txt`
  echo $s
fi

while [ $s -ne -1 ]
do
	eval "./IRIS_3DMM -ipath $DATASET_ROOT_PATH -eye_type 0 -opath $OUTPUT_DIR -mloc model/main_ccnf_wild_70.txt -fdcloc classifiers/haarcascade_frontalface_alt.xml -fmeta $DATASET_META_FILE -clmwild $INPUT_LANDMARK_CMD -con 1"
	if [ -f startF.txt ];
	then
	  s=`cat startF.txt`
          echo $s
	fi
done
