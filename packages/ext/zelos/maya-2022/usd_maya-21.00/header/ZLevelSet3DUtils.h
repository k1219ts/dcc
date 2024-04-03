//--------------------//
// ZLevelSet3DUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZLevelSet3DUtils_h_
#define _ZLevelSet3DUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/**
	Change the given level-set value to the given candidate value if the absolute value of the current level-set is larger that the one of the candidate value.
	@param[in,out] phi The current level-set value being tested.
	@param[in] candidate The candidate level-set value being tested.
	@return True if updated, false otherwise.
*/
inline bool
ZUpdatePhi( float& phi, float candidate )
{
	bool updated = false;
	if( ZAbs(phi) > ZAbs(candidate) ) { phi=candidate; updated=true; }
	return updated;
}

/**
 	Return true if the given state is 'ZFMMState::zInterface' or 'ZFMMState::zUpdated' and false otherwise.
	@param[in] state The state being queried.
 	@return True if the given state is 'ZFMMState::zInterface' or 'ZFMMState::zUpdated' and false otherwise.
*/
inline bool
ZHasPhi( int state )
{
	if( state==ZFMMState::zUpdated   ) { return true; }
	if( state==ZFMMState::zInterface ) { return true; }
	return false;
}

/**
	Return the solution for the quadratic equation from 3 neighbors int x, y, and z-directions.
	The solution for the quadratic equation for 3 neighbors (xyz) is not critical.
	quadratic formula: (phi-phi_x)^2+(phi-phi_y)^2=1
	@note All the candidate values must not be negative.
	@param[in] p The candidate value in x-direction to update from it.
	@param[in] q The candidate value in y-direction to update from it.
	@param[in] r The candidate value in z-direction to update from it.
*/
inline float
ZSolvePhi( float p, float q, float r )
{
	float d = ZMin(p,q,r) + 1; // 1: cell size
	if( d > ZMax(p,q,r) )
	{
		d = ZMin( d, 0.5f*((p+q)+sqrtf(2-ZPow2(p-q))) );
		d = ZMin( d, 0.5f*((q+r)+sqrtf(2-ZPow2(q-r))) );
		d = ZMin( d, 0.5f*((r+p)+sqrtf(2-ZPow2(r-p))) );
	}
	return d;
}

/**
	Given two signed distance values (line end points), determin what fraction of a connecting segment is "inside".
	@param[in] phi0 The first signed distance value.
	@param[in] phi1 The second signed distance value.
	@return The fraction inside the segment.
*/
inline float
ZFractionInside( float phi0, float phi1 )
{
	if( phi0 <  0 && phi1 <  0 ) { return 1; }
	if (phi0 <  0 && phi1 >= 0 ) { return phi0/(phi0-phi1); }
	if( phi0 >= 0 && phi1 <  0 ) { return phi1/(phi1-phi0); }
	return 0;
}

/**
	Given four signed distance values (square corners), determine what fraction of the square is "inside".
	@param[in] phi0 The 1st signed distance value.
	@param[in] phi1 The 2nd signed distance value.
	@param[in] phi2 The 3rd signed distance value.
	@param[in] phi3 The 4th signed distance value.
	@return The fraction inside the square.
*/
float ZFractionInside( float phi0, float phi1, float phi2, float phi3 );

float ZFractionInside( float phi0, float phi1, float phi2, float phi3, float phi4, float phi5, float phi6, float phi7, bool approximation=true );

ZELOS_NAMESPACE_END

#endif

