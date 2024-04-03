//-----------------//
// ZComplexArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.15                               //
//-------------------------------------------------------//

#ifndef _ZComplexArray_h_
#define _ZComplexArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZComplexArray : public ZArray<ZComplex>
{
	private:

		typedef ZArray<ZComplex> parent;

	public:

		ZComplexArray();
		ZComplexArray( const ZComplexArray& a );
		ZComplexArray( int initialLength );
		ZComplexArray( int initialLength, const ZComplex& valueForAll );

//		void add( const float& r, const float& i );
};

//inline void
//ZComplexArray::add( const float& r, const float& i )
//{
//	std::vector<ZComplex>::emplace_back( r, i );
//}

ostream&
operator<<( ostream& os, const ZComplexArray& object );

ZELOS_NAMESPACE_END

#endif

