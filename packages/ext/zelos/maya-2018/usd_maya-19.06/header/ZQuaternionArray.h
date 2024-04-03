//--------------------//
// ZQuaternionArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.03                               //
//-------------------------------------------------------//

#ifndef _ZQuaternionArray_h_
#define _ZQuaternionArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZQuaternionArray : public ZArray<ZQuaternion>
{
	private:

		typedef ZArray<ZQuaternion> parent;

	public:

		ZQuaternionArray();
		ZQuaternionArray( const ZQuaternionArray& a );
		ZQuaternionArray( int initialLength );
		ZQuaternionArray( int initialLength, const ZQuaternion& valueForAll );

//		void add( const float& r, const float& i );
};

//inline void
//ZQuaternionArray::add( const float& r, const float& i )
//{
//	std::vector<ZQuaternion>::emplace_back( r, i );
//}

ostream&
operator<<( ostream& os, const ZQuaternionArray& object );

ZELOS_NAMESPACE_END

#endif

