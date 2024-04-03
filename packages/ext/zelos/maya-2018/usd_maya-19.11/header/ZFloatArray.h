//---------------//
// ZFloatArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2019.03.26                               //
//-------------------------------------------------------//

#ifndef _ZFloatArray_h_
#define _ZFloatArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZFloatArray : public ZArray<float>
{
	private:

		typedef ZArray<float> parent;

	public:

		ZFloatArray();
		ZFloatArray( const ZFloatArray& a );
		ZFloatArray( int initialLength );
		ZFloatArray( int initialLength, float valueForAll );

		void add( float v, bool useOpenMP=false );
		void multiply( float v, bool useOpenMP=false );

		void setRandomValues( int seed=0, float min=0.f, float max=1.f );

        int minIndex() const;
        int maxIndex() const;

		float min( bool useOpenMP=false ) const;
		float max( bool useOpenMP=false ) const;
		float absMax( bool useOpenMP=false ) const;
		void getMinMax( float& min, float& max, bool useOpenMP=false ) const;

		void normalize( bool useOpenMP=false );

		double sum( bool useOpenMP=false ) const;
		double average( bool useOpenMP=false ) const;

		void getAccumulated( ZFloatArray& accumulated ) const;
};

ostream&
operator<<( ostream& os, const ZFloatArray& object );

ZELOS_NAMESPACE_END

#endif

