#!/bin/sh

BASEDIR="$(dirname "$0")"

echo "----- install python module -----"
cp -rf $BASEDIR/pylibs/dxstats /Applications/RV.app/Contents/PlugIns/Python
cp -rf $BASEDIR/pylibs/requests /Applications/RV.app/Contents/PlugIns/Python
cp -rf $BASEDIR/pylibs/pymongo /Applications/RV.app/Contents/PlugIns/Python
cp -rf $BASEDIR/pylibs/xlwt /Applications/RV.app/Contents/PlugIns/Python
cp -rf $BASEDIR/pylibs/bson /Applications/RV.app/Contents/PlugIns/Python
cp -rf $BASEDIR/pylibs/tactic_client_lib /Applications/RV.app/Contents/PlugIns/Python
cp -f $BASEDIR/pylibs/scandir.py /Applications/RV.app/Contents/PlugIns/Python

# install mio_ffmpeg lib
cp -f $BASEDIR/mio_ffmpeg/mojave/mio_ffmpeg.dylib /Applications/RV.app/Contents/PlugIns/MovieFormats

# hotkey patch
cp -f $BASEDIR/patchFile/shotgun_review_app.mu /Applications/RV.app/Contents/PlugIns/Mu

echo "----- install rv packages -----"

# SimpleEdit
/Applications/RV.app/Contents/MacOS/rvpkg -force -remove ~/Library/Application\ Support/RV/Packages/SimpleEdit-1.0.rvpkg
/Applications/RV.app/Contents/MacOS/rvpkg -force -install -add ~/Library/Application\ Support/RV $BASEDIR/rvpkg/SimpleEdit-1.0.rvpkg

# dxSeqLatest
/Applications/RV.app/Contents/MacOS/rvpkg -force -remove ~/Library/Application\ Support/RV/Packages/dxSeqLatest-2.5.rvpkg
/Applications/RV.app/Contents/MacOS/rvpkg -force -remove ~/Library/Application\ Support/RV/Packages/dxSeqLatest-3.0.rvpkg
/Applications/RV.app/Contents/MacOS/rvpkg -force -remove ~/Library/Application\ Support/RV/Packages/dxSeqLatest-3.1.rvpkg
/Applications/RV.app/Contents/MacOS/rvpkg -force -install -add ~/Library/Application\ Support/RV $BASEDIR/rvpkg/dxSeqLatest-3.1.rvpkg

# dxTacticSubmit
/Applications/RV.app/Contents/MacOS/rvpkg -force -remove ~/Library/Application\ Support/RV/Packages/dxTacticSubmit-1.1.rvpkg
/Applications/RV.app/Contents/MacOS/rvpkg -force -remove ~/Library/Application\ Support/RV/Packages/dxTacticSubmit-1.2.rvpkg
/Applications/RV.app/Contents/MacOS/rvpkg -force -install -add ~/Library/Application\ Support/RV $BASEDIR/rvpkg/dxTacticSubmit-1.2.rvpkg

# dxTacticReview
/Applications/RV.app/Contents/MacOS/rvpkg -force -remove ~/Library/Application\ Support/RV/Packages/dxTacticReview-1.1.rvpkg
/Applications/RV.app/Contents/MacOS/rvpkg -force -install -add ~/Library/Application\ Support/RV $BASEDIR/rvpkg/dxTacticReview-1.1.rvpkg

