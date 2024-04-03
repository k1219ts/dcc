#!/bin/bash

BASEDIR=$(dirname $0)
CFDIR=$BASEDIR/../crossFlatform

cd $CFDIR;
rm -rf rv_plugins.zip;
zip -r rv_plugins.zip sources/*;
