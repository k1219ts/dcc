#!/bin/bash
#FFMPEG="ffmpeg" #"/opt/ffmpeg/bin/ffmpeg"
#IMGCVT="imgcvt" #"/usr/local/bin/imgcvt"
#IDENTIFY="identify" #"/usr/bin/identify"
FFMPEG="/opt/ffmpeg/bin/ffmpeg"
IMGCVT="/usr/local/bin/imgcvt"
IDENTIFY="/usr/bin/identify"
#export LD_LIBRARY_PATH=/opt/ffmpeg/lib

startTime=$1
endTime=$2
fps=$3
codec=$4
blastTempDir=$5
fileName=$6
prefix=$7
mode=$8
metadataStr=$9

currentDirectory=`pwd`
cd $blastTempDir
#$IMGCVT -n $startTime $endTime 1 playblast.@@@@.iff $prefix.#.jpg &>/dev/null
# remove iff images
#rm -rf playblast*.iff

if [ $mode = 'sequence' ]; then
  exit 1
fi

x=1; for i in *jpg; do counter=$(printf %04d $x); ln -sf "$i" $prefix."$counter".jpg; x=$(($x+1)); done

format=`$IDENTIFY -format %wx%h $prefix.0001.jpg`
format_array=(`echo $format | tr "x" "\n"`)
width=${format_array[0]}
height=${format_array[1]}

if [ `expr $width % 2` != 0 ]; then
  width=`expr $width - 1`
fi
if [ `expr $height % 2` != 0 ]; then
  height=`expr $height - 1`
fi

cc_opt='-vf mp=eq2=1:1.68:0.3:1.25:1:0.96:1'
cc_opt='-pix_fmt yuv420p'

case $codec in
  "H.264 HQ")
    $FFMPEG -r $fps -i $blastTempDir/$prefix.%04d.jpg -r $fps -an -vcodec libx264 $cc_opt -preset slow -profile:v baseline -b 14000k -tune zerolatency -s ${width}x${height} -metadata title=\'"$metadataStr"\' -y $fileName
    ;;
  "H.264 Normal")
    $FFMPEG -r $fps -i $blastTempDir/$prefix.%04d.jpg -r $fps -an -vcodec libx264 -preset slow -profile:v baseline -b 9000k -tune zerolatency -s ${width}x${height} -metadata title=\'"$metadataStr"\' -y $fileName &>/dev/null
    ;;
  "H.264 LT")
    $FFMPEG -r $fps -i $blastTempDir/$prefix.%04d.jpg -r $fps -an -vcodec libx264 $cc_opt -preset slow -profile:v baseline -b 6000k -tune zerolatency -s ${width}x${height} -metadata title=\'"$metadataStr"\' -y $fileName
    ;;
  "Apple ProRes422 HQ")
    $FFMPEG -r $fps -i $blastTempDir/$prefix.%04d.jpg -r $fps -an -vcodec prores -profile 3 -s ${width}x${height} -metadata title=\'"$metadataStr"\' -y $fileName &>/dev/null
    ;;
  "Apple ProRes422 Normal")
    $FFMPEG -r $fps -i $blastTempDir/$prefix.%04d.jpg -r $fps -an -vcodec prores -profile 2 -s ${width}x${height} -metadata title=\'"$metadataStr"\' -y $fileName &>/dev/null
    ;;
  "Apple ProRes422 LT")
    $FFMPEG -r $fps -i $blastTempDir/$prefix.%04d.jpg -r $fps -an -vcodec prores -profile 1 -s ${width}x${height} -metadata title=\'"$metadataStr"\' -y $fileName &>/dev/null
    ;;
esac


#rm -rf $prefix.*.jpg
cd $currentDirectory
#rm -rf $blastTempDir
