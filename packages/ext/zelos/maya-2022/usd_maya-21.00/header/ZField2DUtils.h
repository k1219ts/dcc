//-----------------//
// ZField2DUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Nayoung Kim @ Dexter Studios                  //
// last update: 2016.09.19                               //
//-------------------------------------------------------//

#ifndef _ZField2DUtils_h_
#define _ZField2DUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

bool Gradient( ZVectorField2D& v, const ZScalarField2D& s, bool useOpenMP=true );
bool Divergence( ZScalarField2D& s, const ZVectorField2D& v, bool useOpenMP=true );

ZPoint TracedPositionByLinear( const ZPoint& startPosition, const ZVector& startVelocity, float dt );
bool ZAdvect( ZVectorField2D& v, const ZVectorField2D& vel, float dt, bool useOpenMP=true );
bool ZAdvect( ZScalarField2D& s, const ZVectorField2D& vel, float dt, bool useOpenMP=true );

void GetVelField( ZVectorField2D& vel, const ZPointArray& vPos, const ZPointArray& vPos0, bool useOpenMP=true );

ZVector ZMinValue( const ZVectorField2D& field, bool useOpenMP=true );
ZVector ZMaxValue( const ZVectorField2D& field, bool useOpenMP=true );
float   ZMinValue( const ZScalarField2D& field, bool useOpenMP=true );
float   ZMaxValue( const ZScalarField2D& field, bool useOpenMP=true );
float   ZMinAbsValue( const ZScalarField2D& field, bool useOpenMP=true );
float   ZMaxAbsValue( const ZScalarField2D& field, bool useOpenMP=true );

bool MapToField( const ZImageMap& img, ZVectorField2D& rgb, ZScalarField2D& cusp, ZScalarField2D& foam, bool useOpenMP=true );
bool MapToVectorField( const ZImageMap& img, ZVectorField2D& field, bool useOpenMP=true );
bool MapToScalarField( const ZImageMap& img, ZScalarField2D& field, int n, bool useOpenMP=true );

bool ArrayToField( const ZVectorArray& arr, ZVectorField2D& field, bool useOpenMP=true );

void VectorFieldLerp( const ZPoint& p, const ZVectorField2D& field, ZVector& vec, float Lx, float Lz );
void VectorFieldInterpolate( const ZPointArray& pos, const ZVectorField2D& field, ZVectorArray& arr, float Lx, float Lz );

void VectorFieldLerpUV( const ZPoint& uv, const ZVectorField2D& field, ZVector& vec, float scaleS=1.f, float scaleT=1.f, float offsetS=0.f, float offsetT=0.f );
void ScalarFieldLerpUV( const ZPoint& uv, const ZScalarField2D& field, float& f, float scaleS=1.f, float scaleT=1.f, float offsetS=0.f, float offsetT=0.f );

void VectorFieldCatromUV( const ZPoint& uv, const ZVectorField2D& field, ZVector& vec, float scaleS=1.f, float scaleT=1.f, float offsetS=0.f, float offsetT=0.f );
void ScalarFieldCatromUV( const ZPoint& uv, const ZScalarField2D& field, float& f, float scaleS=1.f, float scaleT=1.f, float offsetS=0.f, float offsetT=0.f );

void VectorFieldInterpolateUV( const ZPointArray& vUV, const ZVectorField2D& field, ZVectorArray& arr, float scaleS=1.f, float scaleT=1.f, float offsetS=0.f, float offsetT=0.f );
void ScalarFieldInterpolateUV( const ZPointArray& vUV, const ZScalarField2D& field, ZFloatArray& arr, float scaleS=1.f, float scaleT=1.f, float offsetS=0.f, float offsetT=0.f );

ZELOS_NAMESPACE_END

#endif

