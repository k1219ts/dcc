//----------//
// ZPlane.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.29                               //
//-------------------------------------------------------//

#ifndef _ZPlane_h_
#define _ZPlane_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZPlane
{
	private:

		// the four coefficients of the plane equation: ax+by+cz+d=0
		ZVector _normal; // (a,b,c)
		float   _d;      // the distance from the origin

	public:

		ZPlane();
		ZPlane( const ZPlane& plane );
		ZPlane( const float& a, const float& b, const float& c, const float& d );
		ZPlane( const ZPoint& p0, const ZPoint& p1, const ZPoint& p2 );
		ZPlane( const ZPoint& p, const ZVector& unitNormal );

		void reset();

		ZPlane& set( const float& a, const float& b, const float& c, const float& d );
		ZPlane& set( const ZPoint& p0, const ZPoint& p1, const ZPoint& p2 );
		ZPlane& set( const ZPoint& p, const ZVector& unitNormal );

		ZPlane& operator=( const ZPlane& plane );

		void reverse();

		const float& a() const;
		const float& b() const;
		const float& c() const;
		const float& d() const;

		void getCoefficients( float& a, float& b, float& c, float& d ) const;

		const ZVector& normal() const;

		bool isOnThePlane( const ZPoint& p, const float& tolerance=Z_EPS ) const;
		bool isOutside( const ZPoint& p, const float& tolerance=0.f ) const;
		bool isInside( const ZPoint& p, const float& tolerance=0.f ) const;

		float signedDistanceFrom( const ZPoint& p ) const;
		float distanceFrom( const ZPoint& p ) const;

		float signedDistanceFromOrigin() const;
		float distanceFromOrigin() const;

		ZPoint closestPoint( const ZPoint& p ) const;

		void draw() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );
};

ostream& operator<<( ostream& os, const ZPlane& object );

ZELOS_NAMESPACE_END

#endif

