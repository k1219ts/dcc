//---------------------//
// ZBoundingBoxArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZBoundingBoxArray_h_
#define _ZBoundingBoxArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZBoundingBoxArray : public ZArray<ZBoundingBox>
{
	private:

		typedef ZArray<ZBoundingBox> parent;

	public:

		ZBoundingBoxArray();
		ZBoundingBoxArray( const ZBoundingBoxArray& a );
		ZBoundingBoxArray( int initialLength );

		ZBoundingBox boundingBox() const;
};

ostream&
operator<<( ostream& os, const ZBoundingBoxArray& object );

ZELOS_NAMESPACE_END

#endif

