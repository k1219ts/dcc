//------------------//
// ZCurvatureType.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.26                               //
//-------------------------------------------------------//

#ifndef _ZCurvatureType_h_
#define _ZCurvatureType_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZCurvatureType
{
	public:

		enum CurvatureType
		{
			zNone     = 0, ///< none
			zGaussian = 1, ///< Gaussian curvature
			zMean     = 2, ///< mean     curvature
			zIntegral = 3  ///< integral curvature
		};

	public:

		ZCurvatureType() {}

		static ZString name( ZCurvatureType::CurvatureType curvatureType )
		{
			switch( curvatureType )
			{
				default:
				case ZCurvatureType::zNone:     { return ZString("none");     }
				case ZCurvatureType::zGaussian: { return ZString("gaussian"); }
				case ZCurvatureType::zMean:     { return ZString("mean");     }
				case ZCurvatureType::zIntegral: { return ZString("integral"); }
			}
		}
};

inline ostream&
operator<<( ostream& os, const ZCurvatureType& object )
{
	os << "<ZCurvatureType>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

