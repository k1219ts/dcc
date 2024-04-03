//--------------------//
// ZComputingMethod.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.26                               //
//-------------------------------------------------------//

#ifndef _ZComputingMethod_h_
#define _ZComputingMethod_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZComputingMethod
{
	public:

		enum ComputingMethod
		{
			zNone      = 0, ///< none
			zSingleCPU = 1, ///< cell center
			zOpenMP    = 2, ///< node
			zCuda      = 3  ///< face center
		};

	public:

		ZComputingMethod() {}

		static ZString name( ZComputingMethod::ComputingMethod computingMethod )
		{
			switch( computingMethod )
			{
				default:
				case ZComputingMethod::zNone:      { return ZString("none");      }
				case ZComputingMethod::zSingleCPU: { return ZString("singleCPU"); }
				case ZComputingMethod::zOpenMP:    { return ZString("OpenMP");    }
				case ZComputingMethod::zCuda:      { return ZString("Cuda");      }
			}
		}
};

inline ostream&
operator<<( ostream& os, const ZComputingMethod& object )
{
	os << "<ZComputingMethod>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

