//-----------------//
// ZDouble3Array.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.30                               //
//-------------------------------------------------------//

#ifndef _ZDouble3Array_h_
#define _ZDouble3Array_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDouble3Array : public ZArray<ZDouble3>
{
	private:

		typedef ZArray<ZDouble3> parent;

	public:

		ZDouble3Array();
		ZDouble3Array( const ZDouble3Array& a );
		ZDouble3Array( int initialLength );

//		void add( const double& v0, const double& v1, const double& v2 );
};

//inline void
//ZDouble3Array::add( const double& v0, const double& v1, const double& v2 )
//{
//	std::vector<ZDouble3>::emplace_back( v0, v1, v2 );
//}

ostream&
operator<<( ostream& os, const ZDouble3Array& object );

ZELOS_NAMESPACE_END

#endif

