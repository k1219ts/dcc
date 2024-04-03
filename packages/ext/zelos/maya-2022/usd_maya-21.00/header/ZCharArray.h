//--------------//
// ZCharArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZCharArray_h_
#define _ZCharArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZCharArray : public ZArray<char>
{
	private:

		typedef ZArray<char> parent;

	public:

		ZCharArray();
		ZCharArray( const ZCharArray& a );
		ZCharArray( int initialLength );
		ZCharArray( int initialLength, int valueForAll );

		void setMask( const ZCharArray& indices, bool value );
};

ostream&
operator<<( ostream& os, const ZCharArray& object );

ZELOS_NAMESPACE_END

#endif

