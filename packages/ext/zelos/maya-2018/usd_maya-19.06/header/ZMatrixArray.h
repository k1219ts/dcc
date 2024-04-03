//----------------//
// ZMatrixArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZMatrixArray_h_
#define _ZMatrixArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZMatrixArray : public ZArray<ZMatrix>
{
	private:

		typedef ZArray<ZMatrix> parent;

	public:

		ZMatrixArray();
		ZMatrixArray( const ZMatrixArray& a );
		ZMatrixArray( int initialLength );
		ZMatrixArray( int initialLength, const ZMatrix& valueForAll );
};

ostream&
operator<<( ostream& os, const ZMatrixArray& object );

ZELOS_NAMESPACE_END

#endif

