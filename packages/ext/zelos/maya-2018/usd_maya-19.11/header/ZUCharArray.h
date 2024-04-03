//---------------//
// ZUCharArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZUCharArray_h_
#define _ZUCharArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZUCharArray : public ZArray<unsigned char>
{
	private:

		typedef ZArray<unsigned char> parent;

	public:

		ZUCharArray();
		ZUCharArray( const ZUCharArray& a );
		ZUCharArray( int initialLength );
		ZUCharArray( int initialLength, int valueForAll );

		void setMask( const ZUCharArray& indices, bool value );
};

ostream&
operator<<( ostream& os, const ZUCharArray& object );

ZELOS_NAMESPACE_END

#endif

