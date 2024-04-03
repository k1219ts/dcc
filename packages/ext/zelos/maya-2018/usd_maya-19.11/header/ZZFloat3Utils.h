//-----------------//
// ZZFloat3Utils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.16                               //
//-------------------------------------------------------//

#ifndef _ZZFloat3Utils_h_
#define _ZZFloat3Utils_h_

#include <ZelosCudaBase.h>

__device__
inline float3 make_float3( float s )
{
	return make_float3( s, s, s );
}

__device__
inline float3 make_float3( float2 v )
{
	return make_float3( v.x, v.y, 0.0f );
}

__device__
inline float3 make_float3( float2 v, float s )
{
	return make_float3( v.x, v.y, s );
}

__device__
inline float3 make_float3( float4 v )
{
	return make_float3( v.x, v.y, v.z );  // discards w
}

__device__
inline float3 make_float3( const int3& v )
{
	return make_float3( float(v.x), float(v.y), float(v.z) );
}

__device__
inline float3 MIN( const float3& a, const float3& b )
{
	return make_float3( (a.x<b.x)?a.x:b.x, (a.y<b.y)?a.y:b.y, (a.z<b.z)?a.z:b.z );
}

__device__
inline float3 MAX( const float3& a, const float3& b )
{
	return make_float3( (a.x>b.x)?a.x:b.x, (a.y>b.y)?a.y:b.y, (a.z>b.z)?a.z:b.z );
}

__device__
inline float3 operator-( const float3& v )
{
	return make_float3( -v.x, -v.y, -v.z );
}

__device__
inline float3 operator+( const float3& a, const float3& b )
{
	return make_float3( a.x+b.x, a.y+b.y, a.z+b.z );
}

__device__ inline float3 operator+( const float3& a, float b )
{
	return make_float3( a.x+b, a.y+b, a.z+b );
}

__device__
inline float3& operator+=( float3& a, const float3& b )
{
	a.x+=b.x; a.y+=b.y; a.z+=b.z;
	return a;
}

__device__
inline float3 operator-( const float3& a, const float3& b )
{
	return make_float3( a.x-b.x, a.y-b.y, a.z-b.z );
}

__device__
inline float3 operator-( const float3& a, float b )
{
	return make_float3( a.x-b, a.y-b, a.z-b );
}

__device__
inline void operator-=( float3& a, const float3& b )
{
	a.x-=b.x; a.y-=b.y; a.z-=b.z;
}

__device__
inline float3 operator*( const float3& a, const float3& b )
{
	return make_float3( a.x*b.x, a.y*b.y, a.z*b.z );
}

__device__
inline float3 operator*( const float3& v, const float& s )
{
	return make_float3( v.x*s, v.y*s, v.z*s );
}

__device__
inline float3 operator*( float s, const float3& v )
{
	return make_float3( v.x*s, v.y*s, v.z*s );
}

__device__
inline void operator*=( float3& v, float s )
{
	v.x*= s; v.y*=s; v.z*=s;
}

__device__
inline float3 operator/( const float3& v, float s )
{
	const float _d = 1.f / s;
	return make_float3( v.x*_d, v.y*_d, v.z*_d );
}

__device__
inline void operator/=( float3& v, float s )
{
	const float _d = 1.f / s;
	v.x*=_d; v.y*=_d; v.z*=_d;
}

__device__
inline float3 LERP( const float3& a, const float3& b, float t )
{
	const float _t = 1-t;
	return make_float3( _t*a.x+t*b.x, _t*a.y+t*b.y, _t*a.z+t*b.z );
}

__device__
inline float DOT( const float3& a, const float3& b )
{ 
	return (a.x*b.x+a.y*b.y+a.z*b.z);
}

__device__
inline float3 CRS( const float3& a, const float3& b )
{ 
	return make_float3( a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x );
}

__device__
inline float LEN2( const float3& v )
{
	return ( v.x*v.x+v.y*v.y+v.z*v.z );
}

__device__
inline float LEN( const float3& v )
{
	return sqrtf( v.x*v.x+v.y*v.y+v.z*v.z );
}

__device__
inline void NRM( float3& v )
{
	float _d = 1.f / sqrtf( v.x*v.x+v.y*v.y+v.z*v.z );
	v.x*=_d; v.y*=_d; v.z*=_d;
}

#endif

