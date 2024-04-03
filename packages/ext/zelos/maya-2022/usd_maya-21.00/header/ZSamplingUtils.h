//------------------//
// ZSamplingUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZSampleringUtils_h_
#define _ZSampleringUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

int ScatterOnSphere( int number, const ZPoint& center, float radius, int seed, bool asAppending, bool useOpenMP, ZPointArray& samples );

int ScatterInSphere( int number, const ZPoint& center, float radius, int seed, bool asAppending, bool useOpenMP, ZPointArray& samples );

int ScatterPoissonDisk2D( float radius, const ZPoint& minPt, const ZPoint& maxPt, int whichPlane, int seed, bool asAppending, ZPointArray& sample );

int ScatterPoissonDisk3D( float radius, const ZPoint& minPt, const ZPoint& maxPt, int seed, bool asAppending, ZPointArray& sample );

ZELOS_NAMESPACE_END

#endif

