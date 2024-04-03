//-----------------//
// ZDouble2Array.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.30                               //
//-------------------------------------------------------//

#ifndef _ZDouble2Array_h_
#define _ZDouble2Array_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDouble2Array : public ZArray<ZDouble2>
{
	private:

		typedef ZArray<ZDouble2> parent;

	public:

		ZDouble2Array();
		ZDouble2Array( const ZDouble2Array& a );
		ZDouble2Array( int initialLength );

//		void add( const double& v0, const double& v1 );
};

//inline void
//ZDouble2Array::add( const double& v0, const double& v1 )
//{
//	std::vector<ZDouble2>::emplace_back( v0, v1 );
//}

ostream&
operator<<( ostream& os, const ZDouble2Array& object );

ZELOS_NAMESPACE_END

#endif

