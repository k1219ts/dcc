//----------------//
// ZFloat3Array.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.15                               //
//-------------------------------------------------------//

#ifndef _ZFloat3Array_h_
#define _ZFloat3Array_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZFloat3Array : public ZArray<ZFloat3>
{
	private:

		typedef ZArray<ZFloat3> parent;

	public:

		ZFloat3Array();
		ZFloat3Array( const ZFloat3Array& a );
		ZFloat3Array( int initialLength );

//		void add( const float& v0, const float& v1, const float& v2 );
};

//inline void
//ZFloat3Array::add( const float& v0, const float& v1, const float& v2 )
//{
//	std::vector<ZFloat3>::emplace_back( v0, v1, v2 );
//}

ostream&
operator<<( ostream& os, const ZFloat3Array& object );

ZELOS_NAMESPACE_END

#endif

