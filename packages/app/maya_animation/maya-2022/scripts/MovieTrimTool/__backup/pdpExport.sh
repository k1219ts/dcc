#!/bin/bash

pdplayerPath="/opt/pdplayer/pdplayer/pdplayerqt64"
#pdplayerPath="/usr/local/bin/pdplayer"

export VRAY_AUTH_CLIENT_FILE_PATH=/netapp/backstage/pub/apps/pdplayer/license

movPath=$1

outPath=$2
outPut="$outPath"/img.0001.jpg
StartFrame=$3
EndFrame=$4

format=`ffmpeg -i "$movPath" 2>&1 | grep Stream | grep -oP ', \K[0-9]+x[0-9]+'`
format_array=(`echo $format | tr "x" "\n"`)
movSizeW=${format_array[0]}
movSizeH=${format_array[1]}

if [ `expr $movSizeW % 2` != 0 ]; then
  movSizeW=`expr $movSizeW - 1`
fi
if [ `expr $movSizeH % 2` != 0 ]; then
  movSizeH=`expr $movSizeH - 1`
fi

$pdplayerPath "$movPath" --mask_size=$movSizeW,$movSizeH --in_point=$StartFrame --out_point=$EndFrame --wa_begin=$StartFrame --wa_end=$EndFrame --save_mask_as_sequence="$outPut" --exit &>/dev/null
