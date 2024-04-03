//-------------//
// ZIntArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.11.01                               //
//-------------------------------------------------------//

#ifndef _ZIntArray_h_
#define _ZIntArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZIntArray : public ZArray<int>
{
	private:

		typedef ZArray<int> parent;

	public:

		ZIntArray();
		ZIntArray( const ZIntArray& a );
		ZIntArray( int initialLength );
		ZIntArray( int initialLength, int valueForAll );

		void serialize( int startIndex=0 );

		void setRandomValues( int seed=0, int min=1, int max=100 );

		void add( int numberToBeAdded );

		int min( bool useOpenMP=false ) const;
		int max( bool useOpenMP=false ) const;
		int absMax( bool useOpenMP=false ) const;
		void getMinMax( int& min, int& max, bool useOpenMP=false ) const;

		double sum( bool useOpenMP=false ) const;
		double average( bool useOpenMP=false ) const;

		void getAccumulated( ZIntArray& accumulated ) const;

		void setMask( const ZIntArray& indices, bool value );
};

ostream&
operator<<( ostream& os, const ZIntArray& object );

ZELOS_NAMESPACE_END

#endif

