//----------------//
// ZVectorArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2019.03.26                               //
//-------------------------------------------------------//

#ifndef _ZVectorArray_h_
#define _ZVectorArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZVectorArray : public ZArray<ZVector>
{
	private:

		typedef ZArray<ZVector> parent;

	public:

		ZVectorArray();
		ZVectorArray( const ZVectorArray& a );
		ZVectorArray( int initialLength );
		ZVectorArray( int initialLength, const ZVector& valueForAll );

//		void add( const float& x, const float& y, const float& z );

		// valid only for points
		ZPoint center() const;
		ZBoundingBox boundingBox( bool useOpenMP=true ) const;
		void addPoints( int numPoints, const ZPoint& minPoint, const ZPoint& maxPoint, int seed=0 );
		void getCovarianceMatrix( ZMatrix& covarianceMatrix ) const;

		void drawPoints( bool useRandomColor=false ) const;
		void drawVectors( const ZVectorArray& position, float scale, bool useRandomColor=false ) const;

		// valid only for vectors (not points)
		float maxMagnitude() const;

        int minMagnitudeIndex() const;
        int maxMagnitudeIndex() const;

		void scale( float v, bool useOpenMP=false );

		void applyTransform( const ZMatrix& matrix, bool asVector, bool useOpenMP=true );
};

//inline void
//ZVectorArray::add( const float& x, const float& y, const float& z )
//{
//	std::vector<ZVector>::emplace_back( x, y, z );
//}

ostream&
operator<<( ostream& os, const ZVectorArray& object );

typedef ZVectorArray ZPointArray;

void PrintLocator( const ZPointArray& ptc );

ZELOS_NAMESPACE_END

#endif
 
