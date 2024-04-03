//------------------//
// ZRotationOrder.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.26                               //
//-------------------------------------------------------//

#ifndef _ZRotationOrder_h_
#define _ZRotationOrder_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// Maya's default rotation order is kXYZ.
// : q^T = p^T * (Rx * Ry * Rz);
// In Zelos, the corresponding order is zZYX,
// because Zelos uses row-major ordering matrix system while Maya uses colume-major ordering matrix system.
// (One is the transpose matrix of the other.)
// Therefore, after applying a rotation matrix,
// Zelos: counter-clockwise rotation (i.e. right handed coordinate system)
// Maya: clockwise rotation (i.e. left handed coordinate system)

// For the same effect as Maya,
// Use zZYX, and then transpose the matrix.

class ZRotationOrder
{
	public:

		enum RotationOrder
		{
			zXYZ = 0,	// q = (XYZ)p = X(Y(Zp)): Z->Y->X
			zYZX = 1,	// q = (YZX)p = Y(Z(Xp)): X->Z->Y
			zZXY = 2,	// q = (ZXY)p = Z(X(Yp)): Y->X->Z
			zXZY = 3,	// q = (XZY)p = X(Z(Yp)): Y->Z->X
			zYXZ = 4,	// q = (YXZ)p = Y(X(Zp)): Z->X->Y
			zZYX = 5	// q = (ZYX)p = Z(Y(Xp)): X->y->Z
		};

	public:

		ZRotationOrder() {}

		static ZString name( ZRotationOrder::RotationOrder rotationOrder )
		{
			switch( rotationOrder )
			{
				default:
				case ZRotationOrder::zXYZ: { return ZString("zXYZ"); }
				case ZRotationOrder::zYZX: { return ZString("zYZX"); }
				case ZRotationOrder::zZXY: { return ZString("zZXY"); }
				case ZRotationOrder::zXZY: { return ZString("zXZY"); }
				case ZRotationOrder::zYXZ: { return ZString("zYXZ"); }
				case ZRotationOrder::zZYX: { return ZString("zZYX"); }
			}
		}
};

inline ostream&
operator<<( ostream& os, const ZRotationOrder& object )
{
	os << "<ZRotationOrder>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

