//----------------//
// ZDoubleArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.24                               //
//-------------------------------------------------------//

#ifndef _ZDoubleArray_h_
#define _ZDoubleArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDoubleArray : public ZArray<double>
{
	private:

		typedef ZArray<double> parent;

	public:

		ZDoubleArray();
		ZDoubleArray( const ZDoubleArray& a );
		ZDoubleArray( int initialLength );
		ZDoubleArray( int initialLength, double valueForAll );

		void add( double v, bool useOpenMP=false );
		void multiply( double v, bool useOpenMP=false );

		void setRandomValues( int seed=0, double min=0.0, double max=1.0 );

		double min( bool useOpenMP=false ) const;
		double max( bool useOpenMP=false ) const;
		double absMax( bool useOpenMP=false ) const;
		void getMinMax( double& min, double& max, bool useOpenMP=false ) const;

		void normalize( bool useOpenMP=false );

		double sum( bool useOpenMP=false ) const;
		double average( bool useOpenMP=false ) const;

		void getAccumulated( ZDoubleArray& accumulated ) const;
};

ostream&
operator<<( ostream& os, const ZDoubleArray& object );

ZELOS_NAMESPACE_END

#endif

