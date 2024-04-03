//-------------------//
// ZSamplingMethod.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.26                               //
//-------------------------------------------------------//

#ifndef _ZSamplingMethod_h_
#define _ZSamplingMethod_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// zRandomBarycentric1
// 1. Determine the sampling count per each triangle by considering its area multiplied by density value at the triangle center.
//    (With no given density map, each triangle area would be multiplied by 1.)
// 2. Generate a barycentric coordinate using two random numbers per each sampling point.
// 3. As a result, the final sampling count matches the given "targetNumber" with or without the given density map.

// zRandomBarycentric2
// 1. Determine the sampling count per each triangle by considering its area.
//    (Similar to "zRandomBarycentric1", but different in that this scheme doesn't consider density map at this phase.)
// 2. Generate a barycentric coordinate using two random numbers per each sampling point.
// 3. Delete sampling points stochastically by considering density value at their UV coordinates.
//    (No given density map, no sampling points deletion)
// 4. As a result, the final sampling count is less than the given "targetNumber" if a density map are given.
//    (Without density map, it matches the given "targetNumber".)

// zMonteCarlo
// based on "Efficient and Flexible Sampling with Blue Noise Properties of Triangle Meshes".
// 0. "zRandomBarycentric1" and "zRandomBarycentric2" generate sampling points on triangles in triangle index order.
//    So, if "targetNumber" is less than the number of triangles, it produces an bias solution because there are no sampling points on the triangles with high indices.
// 1. This scheme uses the probability, which is proportional to the triangle area, rather than index order.
// 2. Delete sampling points stochastically by considering density value at their UV coordinates.
//    (No given density map, no sampling points deletion)
// 3. As a result, the final sampling count is less than the given "targetNumber" if a density map are given.
//    (Without density map, it matches the given "targetNumber".)

// zPoissonDisk
// based on "Efficient and Flexible Sampling with Blue Noise Properties of Triangle Meshes".
// 0. Poisson disk sampling scheme with blue noise properties on the triangle mesh
// 1. Generate pre-sampling based on "zMonteCarlo" scheme.
// 2. Delete sampling points inside the given radius from any other points.
// 3. As a result, the final sampling count is less than the given "targetNumber" if a density map are given and the radius is unique.

class ZSamplingMethod
{
	public:

		enum SamplingMethod
		{
			zNone                = 0,
			zRandomBarycentric1  = 1,
			zRandomBarycentric2  = 2,
			zPoissonDiskOnUV     = 3,
			zMonteCarlo          = 4,
			zPoissonDiskOnMesh   = 5,
			zWisp                = 6
		};

	public:

		ZSamplingMethod() {}

		static ZString name( ZSamplingMethod::SamplingMethod samplingMethod )
		{
			switch( samplingMethod )
			{
				default:
				case ZSamplingMethod::zNone:               { return ZString("none");                 }
				case ZSamplingMethod::zRandomBarycentric1: { return ZString("random barycentric 1"); }
				case ZSamplingMethod::zRandomBarycentric2: { return ZString("random barycentric 2"); }
				case ZSamplingMethod::zPoissonDiskOnUV:    { return ZString("Poisson disk on UV");   }
				case ZSamplingMethod::zMonteCarlo:         { return ZString("Monte Carlo");          }
				case ZSamplingMethod::zPoissonDiskOnMesh:  { return ZString("Poisson disk on mesh"); }
				case ZSamplingMethod::zWisp:               { return ZString("Wisp");                 }
			}
		}
};

inline ostream&
operator<<( ostream& os, const ZSamplingMethod& object )
{
	os << "<ZSamplingMethod>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

