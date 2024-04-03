#!/bin/bash

BASEDIR=$(dirname $0)
ROOTDIR=$(dirname $0)/..
# 
PKGDIR=$ROOTDIR/PackageFiles
DISTDIR=$ROOTDIR/Packages
# 
MUDIR=$ROOTDIR/Mu
PYTHONDIR=$ROOTDIR/Python

function pkg {
    local pkg=$1
    local ver=$2
    local files=${@:3}

    echo $pkg;
    echo $ver;
    echo $files;

    zip -j $DISTDIR/$pkg-$ver.rvpkg $PKGDIR/$pkg/PACKAGE $files;
    cp -arv $DISTDIR/$pkg-$ver.rvpkg $ROOTDIR/crossFlatform/sources/rvpkg/
}

# dxOTIO
pkg dxOTIO 2.3 \
$MUDIR/gotoInput.mu \
$PYTHONDIR/otio_reader.py \
$PYTHONDIR/otio_reader_plugin.py \
$PYTHONDIR/overlay_hud.py \
;

# dxSeqLatest
pkg dxSeqLatest 3.1 \
$MUDIR/dxTask_selector.mu \
$PYTHONDIR/dxConfig.py \
$PYTHONDIR/dxSeqLatest.py \
$PYTHONDIR/ui_dxSeqLatest.py \
$PYTHONDIR/dxEditOrder.py \
$PYTHONDIR/ui_dxEditOrder.py \
$PYTHONDIR/dxTacticCommon.py \
$PYTHONDIR/dxVersioning.py \
$PYTHONDIR/dxVersioning_api.py \
;

# dxTacticReview
pkg dxTacticReview 1.1 \
$PYTHONDIR/dxTacticReview.py \
$PYTHONDIR/dxTacticCommon.py \
$PYTHONDIR/dxTacticWidget.py \
$PYTHONDIR/ui_dxTacticReview.py \
;

# dxTacticSubmit
pkg dxTacticSubmit 1.2 \
$PYTHONDIR/dxConfig.py \
$PYTHONDIR/dxTacticSubmit.py \
$PYTHONDIR/ui_tacticSubmit.py \
$PYTHONDIR/dxTacticCommon.py \
;

# SimpleEdit
pkg SimpleEdit 1.0 \
$PYTHONDIR/simple_edit.py \
;

# dxOCIO
pkg dxOCIO 2.3 \
$PYTHONDIR/ocio_source_setup.py \
;

# dxRenameEditOrder
pkg dxRenameEditOrder 1.0 \
$PYTHONDIR/dxRenameEditOrder.py \
$PYTHONDIR/ui_dxRenameEditOrder.py \
;