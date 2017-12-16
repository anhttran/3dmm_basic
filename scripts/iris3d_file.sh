#!/bin/bash
INPUT_FILE_PATH='../data/images/avgface.jpg';
OUTPUT_DIR_PATH='../output/avgface.ply';
eval "./IRIS_3DMM -f $INPUT_FILE_PATH -of $OUTPUT_DIR_PATH -mloc model/main_ccnf_general.txt -fdcloc classifiers/haarcascade_frontalface_alt.xml"
