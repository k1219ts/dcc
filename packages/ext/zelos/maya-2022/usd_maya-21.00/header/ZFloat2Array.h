//----------------//
// ZFloat2Array.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.15                               //
//-------------------------------------------------------//

#ifndef _ZFloat2Array_h_
#define _ZFloat2Array_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZFloat2Array : public ZArray<ZFloat2>
{
	private:

		typedef ZArray<ZFloat2> parent;

	public:

		ZFloat2Array();
		ZFloat2Array( const ZFloat2Array& a );
		ZFloat2Array( int initialLength );

//		void add( const float& v0, const float& v1 );
};

//inline void
//ZFloat2Array::add( const float& v0, const float& v1 )
//{
//	std::vector<ZFloat2>::emplace_back( v0, v1 );
//}

ostream&
operator<<( ostream& os, const ZFloat2Array& object );

ZELOS_NAMESPACE_END

#endif

