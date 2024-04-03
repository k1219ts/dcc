//--------//
// ZRay.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.29                               //
//-------------------------------------------------------//

#ifndef _ZRay_h_
#define _ZRay_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZRay
{
	private:

		ZPoint  _origin;
		ZVector _direction;

		float   _min;
		float   _max;

	public:

		ZRay();
		ZRay( const ZRay& ray );
		ZRay( const ZPoint& origin, const ZVector& direction, const float& min=0.f, const float& max=Z_FLTMAX );

		void reset();

		ZRay& set( const ZPoint& origin, const ZVector& direction, const float& min=0.f, const float& max=Z_FLTMAX );

		ZRay& operator=( const ZRay& ray );

		void getPoint( const float& t, ZPoint& p ) const;
		ZPoint point( const float& t ) const;

		const ZPoint& origin() const;
		const ZVector& direction() const;

		const float& min() const;
		const float& max() const;

		void draw() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );
};

ostream& operator<<( ostream& os, const ZRay& object );

ZELOS_NAMESPACE_END

#endif

