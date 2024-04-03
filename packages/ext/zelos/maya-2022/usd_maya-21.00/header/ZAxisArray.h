//--------------//
// ZAxisArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZAxisArray_h_
#define _ZAxisArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZAxisArray : public ZArray<ZAxis>
{
	private:

		typedef ZArray<ZAxis> parent;

	public:

		ZAxisArray();
		ZAxisArray( const ZAxisArray& a );
		ZAxisArray( int initialLength );

		void changeHandedness( int i );

		void offset( int whichAxis, float e );

		void drawOrigins() const;
		void draw( float scale, bool bySimpleLine=true ) const;
};

ostream&
operator<<( ostream& os, const ZAxisArray& object );

ZELOS_NAMESPACE_END

#endif

