//----------------//
// ZBoundingBox.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.29                               //
//-------------------------------------------------------//

#ifndef _ZBoundingBox_h_
#define _ZBoundingBox_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A 3D axis-aligned bounding box.
/**
	A bounding box describes a volume in world space that bounds a piece of geometry.
	The box is defined by two corner points which describe the minimum and maximum of the box.
*/
class ZBoundingBox
{
	private:

		bool   _initialized;

		ZPoint _min;			///< the minimum corner point
		ZPoint _max;			///< the maximum corner point

	public:

		ZBoundingBox();
		ZBoundingBox( const ZBoundingBox& b );
		ZBoundingBox( const ZPoint& p1, const ZPoint& p2 );
		ZBoundingBox( const ZFloat3& p1, const ZFloat3& p2 );
		ZBoundingBox( const ZDouble3& p1, const ZDouble3& p2 );
		ZBoundingBox( const ZPoint& p1, const ZPoint& p2, const ZPoint& p3 );
		ZBoundingBox( const ZFloat3& p1, const ZFloat3& p2, const ZFloat3& p3 );
		ZBoundingBox( const ZDouble3& p1, const ZDouble3& p2, const ZDouble3& p3 );

		void reset();

		ZBoundingBox& set( const ZPoint& p1, const ZPoint& p2 );
		ZBoundingBox& set( const ZFloat3& p1, const ZFloat3& p2 );
		ZBoundingBox& set( const ZDouble3& p1, const ZDouble3& p2 );
		ZBoundingBox& set( const ZPoint& p1, const ZPoint& p2, const ZPoint& p3 );
		ZBoundingBox& set( const ZFloat3& p1, const ZFloat3& p2, const ZFloat3& p3 );
		ZBoundingBox& set( const ZDouble3& p1, const ZDouble3& p2, const ZDouble3& p3 );

		ZBoundingBox& operator=( const ZBoundingBox& source );

		ZBoundingBox& operator*=( const int& scale );
		ZBoundingBox& operator*=( const float& scale );
		ZBoundingBox& operator*=( const double& scale );

		ZBoundingBox& expand( const ZPoint& p );
		ZBoundingBox& expand( const ZFloat3& p );
		ZBoundingBox& expand( const ZDouble3& p );
		ZBoundingBox& expand( const ZBoundingBox& box );
		ZBoundingBox& expand( float epsilon=Z_EPS );

		ZBoundingBox& scaleAboutCenter( const ZVector& scale );
		ZBoundingBox& scale( const ZVector& scale, const ZPoint& pivot );

		ZBoundingBox& move( const ZVector& translation );

		ZBoundingBox& merge( const ZBoundingBox& b0, const ZBoundingBox& b1 );

		bool initialized() const;

		bool contains( const ZPoint& point ) const; 
		bool contains( const ZFloat3& p ) const; 
		bool contains( const ZDouble3& p ) const; 

		bool intersects( const ZBoundingBox& box ) const;
		bool intersectsWithLineSegment( const ZPoint& a, const ZPoint& b ) const;
		bool intersectsWithRay( const ZPoint& rayOrigin, const ZVector& rayDirection, ZPoint* nearPoint=NULL, ZPoint* farPoint=NULL, float epsilon=Z_EPS ) const;
		bool intersectsWithTriangle( const ZPoint& a, const ZPoint& b, const ZPoint& c ) const;
		bool intersectsWithSphere( const ZPoint& center, float radius ) const;

		float distanceFromOutside( const ZPoint& p, bool asSquaredDist=false ) const;

		void split( ZBoundingBox& child1, ZBoundingBox& child2 ) const;

		void offset( const ZVector& displacement );

		const ZPoint& minPoint() const { return _min; }
		const ZPoint& maxPoint() const { return _max; }

		ZPoint center() const;

		float xWidth() const;
		float yWidth() const;
		float zWidth() const;

		float width( int dimension ) const;

		float maxWidth() const;
		float minWidth() const;

		float volume() const;
		float diagonalLength() const;

		int maxDimension() const;

		void getBoundingSphere( ZPoint& center, float& radius ) const;

		void getEightCorners( ZPoint& p0, ZPoint& p1, ZPoint& p2, ZPoint& p3, ZPoint& p4, ZPoint& p5, ZPoint& p6, ZPoint& p7 ) const;

		void getEightCorners( ZPoint p[8] ) const;

		void applyTransform( const ZMatrix& xform );

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		void draw() const;
		void drawVertices() const;
		void drawWireframe() const;
		void drawSurface( bool withNormal=false ) const;
		void drawWireSurface( const ZColor& wireColor=ZColor::gray(), bool withNormal=false ) const;
};

ostream& operator<<( ostream& os, const ZBoundingBox& object );

ZELOS_NAMESPACE_END

#endif

