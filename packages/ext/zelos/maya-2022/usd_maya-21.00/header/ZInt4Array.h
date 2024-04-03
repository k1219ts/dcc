//--------------//
// ZInt4Array.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.18                               //
//-------------------------------------------------------//

#ifndef _ZInt4Array_h_
#define _ZInt4Array_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZInt4Array : public ZArray<ZInt4>
{
	private:

		typedef ZArray<ZInt4> parent;

	public:

		ZInt4Array();
		ZInt4Array( const ZInt4Array& a );
		ZInt4Array( int initialLength );

//		void add( const int& v0, const int& v1, const int& v2, const int& v3 );
};

//inline void
//ZInt4Array::add( const int& v0, const int& v1, const int& v2, const int& v3 )
//{
//	std::vector<ZInt4>::emplace_back( v0, v1, v2, v3 );
//}

ostream&
operator<<( ostream& os, const ZInt4Array& object );

ZELOS_NAMESPACE_END

#endif

