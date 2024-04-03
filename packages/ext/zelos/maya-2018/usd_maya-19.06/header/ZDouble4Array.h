//-----------------//
// ZDouble4Array.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.30                               //
//-------------------------------------------------------//

#ifndef _ZDouble4Array_h_
#define _ZDouble4Array_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDouble4Array : public ZArray<ZDouble4>
{
	private:

		typedef ZArray<ZDouble4> parent;

	public:

		ZDouble4Array();
		ZDouble4Array( const ZDouble4Array& a );
		ZDouble4Array( int initialLength );

//		void add( const double& v0, const double& v1, const double& v2, const double& v3 );
};

//inline void
//ZDouble4Array::add( const double& v0, const double& v1, const double& v2, const double& v3 )
//{
//	std::vector<ZDouble4>::emplace_back( v0, v1, v2, v3 );
//}

ostream&
operator<<( ostream& os, const ZDouble4Array& object );

ZELOS_NAMESPACE_END

#endif

