//----------------//
// ZFloat4Array.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.15                               //
//-------------------------------------------------------//

#ifndef _ZFloat4Array_h_
#define _ZFloat4Array_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZFloat4Array : public ZArray<ZFloat4>
{
	private:

		typedef ZArray<ZFloat4> parent;

	public:

		ZFloat4Array();
		ZFloat4Array( const ZFloat4Array& a );
		ZFloat4Array( int initialLength );

//		void add( const float& v0, const float& v1, const float& v2, const float& v3 );
};

//inline void
//ZFloat4Array::add( const float& v0, const float& v1, const float& v2, const float& v3 )
//{
//	std::vector<ZFloat4>::emplace_back( v0, v1, v2, v3 );
//}

ostream&
operator<<( ostream& os, const ZFloat4Array& object );

ZELOS_NAMESPACE_END

#endif

