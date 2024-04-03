#!/bin/bash
FFMPEG="ffmpeg" #"/opt/ffmpeg/bin/ffmpeg"
IMGCVT="imgcvt" #"/usr/local/bin/imgcvt"
IDENTIFY="identify" #"/usr/bin/identify"
#export LD_LIBRARY_PATH=/opt/ffmpeg/lib

startTime=$1
endTime=$2
fps=$3
codec=$4
blastTempDir="$5"
fileName="$6"
prefix="$7"
mode=$8

currentDirectory=`pwd`
cd "$blastTempDir"

x=1; for i in *jpg; do counter=$(printf %04d $x); ln -sf "$i" img"$counter".jpg; x=$(($x+1)); done

format=`$IDENTIFY -format %wx%h "$blastTempDir"/img.0001.jpg`
format_array=(`echo $format | tr "x" "\n"`)
width=${format_array[0]}
height=${format_array[1]}

if [ `expr $width % 2` != 0 ]; then
  width=`expr $width - 1`
fi
if [ `expr $height % 2` != 0 ]; then
  height=`expr $height - 1`
fi


case $codec in
  "H.264 HQ")
    $FFMPEG -r $fps -i $blastTempDir/img%04d.jpg -r $fps -an -vcodec libx264 -preset slow -profile:v baseline -b 14000k -tune zerolatency -s ${width}x${height} -y $fileName &>/dev/null    
    ;;
  "H.264 Normal")
    $FFMPEG -r $fps -i $blastTempDir/img%04d.jpg -r $fps -an -vcodec libx264 -preset slow -profile:v baseline -b 9000k -tune zerolatency -s ${width}x${height} -y $fileName &>/dev/null
    ;;
  "H.264 LT")
    $FFMPEG -r $fps -i "$blastTempDir/img%04d.jpg" -r $fps -an -vcodec libx264 -preset slow -profile:v baseline -b 6000k -tune zerolatency -s ${width}x${height} -y "$fileName" &>/dev/null
    ;;
  "Apple ProRes422 HQ")
    $FFMPEG -r $fps -i $blastTempDir/img%04d.jpg -r $fps -an -vcodec prores -profile 3 -s ${width}x${height} -y $fileName &>/dev/null
    ;;
  "Apple ProRes422 Normal")
    $FFMPEG -r $fps -i $blastTempDir/img%04d.jpg -r $fps -an -vcodec prores -profile 2 -s ${width}x${height} -y $fileName &>/dev/null
    ;;
  "Apple ProRes422 LT")
    $FFMPEG -r $fps -i $blastTempDir/img%04d.jpg -r $fps -an -vcodec prores -profile 1 -s ${width}x${height} -y $fileName &>/dev/null
    ;;
esac


rm -rf img*.jpg
cd $currentDirectory
rm -rf "$blastTempDir"
