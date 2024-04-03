//--------------//
// ZInt2Array.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.18                               //
//-------------------------------------------------------//

#ifndef _ZInt2Array_h_
#define _ZInt2Array_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZInt2Array : public ZArray<ZInt2>
{
	private:

		typedef ZArray<ZInt2> parent;

	public:

		ZInt2Array();
		ZInt2Array( const ZInt2Array& a );
		ZInt2Array( int initialLength );

//		void add( const int& v0, const int& v1 );
};

//inline void
//ZInt2Array::add( const int& v0, const int& v1 )
//{
//	std::vector<ZInt2>::emplace_back( v0, v1 );
//}

ostream&
operator<<( ostream& os, const ZInt2Array& object );

ZELOS_NAMESPACE_END

#endif

