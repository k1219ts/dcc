//--------------------//
// ZQuaternionUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZQuaternionUtils_h_
#define _ZQuaternionUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief The spherical linear interpolation between two quaternions.
/**
	As t goes from 0 to 1, the quaternion returned goes from q1 to q2.
	@param[in] q1 The quaternion to rotate from.
	@param[in] q2 The quaternion to rotate to.
	@param[in] t The interpolation value.
	@param[in] tolerance The tolerance.
	@return The quaternion that has been interpolated from q1 to q2.
*/
ZQuaternion Slerp( const ZQuaternion& q1, const ZQuaternion& q2, float t, float tolerance=0.00001f );

/// @brief The spherical linear interpolation between two rotation matrix.
/**
	As t goes from 0 to 1, the rotation matrix returned goes from m1 to m2.
	The interpolation always takes shortest path (in quaternion space) from m1 to m2.
	@param[in] m1 The rotation matrix to rotate from.
	@param[in] m2 The rotation matrix to rotate to.
	@param[in] t The interpolation value.
	@return The interpolated rotation matrix.
*/
ZMatrix Slerp( const ZMatrix& m1, const ZMatrix& m2, float t );

/**
	Interpolate between m1 and m4 by converting m1 ... m4 into quaternions and treating them as control points of a Bezier curve using slerp in place of lerp in the De Castlejeau evaluation algorithm.
	Just like a cubic Bezier curve, this will interpolate m1 at t=0 and m4 at t=1 but in general will not pass through m2 and m3.
	Unlike a standard Bezier curve this curve will not have the convex hull property.
	@note m1 ... m4 must be rotation matrices!
	@param[in] m1 The 1st rotation matrix.
	@param[in] m2 The 2nd rotation matrix.
	@param[in] m3 The 3rd rotation matrix.
	@param[in] m4 The 4th rotation matrix.
	@param[in] t The interpolation value.
	@return The interpolated rotation matrix.
*/
ZMatrix BezLerp( const ZMatrix& m1, const ZMatrix& m2, const ZMatrix& m3, const ZMatrix& m4, float t );

ZELOS_NAMESPACE_END

#endif

