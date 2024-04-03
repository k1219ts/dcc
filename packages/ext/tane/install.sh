#!/bin/bash

echo "Install Tane resource"

target='/netapp/backstage/pub/lib/Tane/1.0.0702/19.05/resource'
if [ ! -d $target/baseGl ]; then
    echo "install ok"
    mkdir -p $target
    cp -rv ${REZ_TANE_ROOT}/resource/baseGl $target
fi
