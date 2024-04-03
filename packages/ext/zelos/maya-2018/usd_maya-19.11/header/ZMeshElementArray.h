//---------------------//
// ZMeshElementArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZMeshElementArray_h_
#define _ZMeshElementArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZMeshElementArray : public ZArray<ZMeshElement>
{
	private:

		typedef ZArray<ZMeshElement> parent;

	public:

		ZMeshElementArray();
		ZMeshElementArray( const ZMeshElementArray& a );
		ZMeshElementArray( int initialLength );
};

ostream&
operator<<( ostream& os, const ZMeshElementArray& object );

ZELOS_NAMESPACE_END

#endif

