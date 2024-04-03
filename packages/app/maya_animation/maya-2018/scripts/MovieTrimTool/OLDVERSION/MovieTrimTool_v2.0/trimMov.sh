#!/bin/bash
FFMPEG="ffmpeg" #"/opt/ffmpeg/bin/ffmpeg"
IMGCVT="imgcvt" #"/usr/local/bin/imgcvt"
IDENTIFY="identify" #"/usr/bin/identify"
#export LD_LIBRARY_PATH=/opt/ffmpeg/lib

startTime=$1
duration=$2
fileName="$3"
movName="$4"
fps=$5

#fps=`ffmpeg -i $movName 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p"`

$FFMPEG -i "$movName" -ss $startTime -t $duration -vcodec libx264 -acodec copy -preset slow -profile:v baseline -b 6000k -tune zerolatency -y "$fileName" &>/dev/null

