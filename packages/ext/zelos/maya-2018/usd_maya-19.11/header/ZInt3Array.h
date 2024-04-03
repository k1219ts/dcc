//--------------//
// ZInt3Array.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.18                               //
//-------------------------------------------------------//

#ifndef _ZInt3Array_h_
#define _ZInt3Array_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZInt3Array : public ZArray<ZInt3>
{
	private:

		typedef ZArray<ZInt3> parent;

	public:

		ZInt3Array();
		ZInt3Array( const ZInt3Array& a );
		ZInt3Array( int initialLength );

//		void add( const int& v0, const int& v1, const int& v2 );
};

//inline void
//ZInt3Array::add( const int& v0, const int& v1, const int& v2 )
//{
//	std::vector<ZInt3>::emplace_back( v0, v1, v2 );
//}

ostream&
operator<<( ostream& os, const ZInt3Array& object );

ZELOS_NAMESPACE_END

#endif

