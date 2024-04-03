//-------------//
// ZTriangle.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.29                               //
//-------------------------------------------------------//

#ifndef _ZTriangle_h_
#define _ZTriangle_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZTriangle
{
	private:

		ZPoint _a;
		ZPoint _b;
		ZPoint _c;

	public:

		ZTriangle();
		ZTriangle( const ZTriangle& triangle );
		ZTriangle( const ZPoint& a, const ZPoint& b, const ZPoint& c );

		void reset();

		ZTriangle& set( const ZPoint& a, const ZPoint& b, const ZPoint& c );

		ZTriangle& operator=( const ZTriangle& triangle );

		const ZPoint& a() const;
		const ZPoint& b() const;
		const ZPoint& c() const;

		ZPoint center() const;

		float area() const;

		void draw( bool shaded=true ) const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );
};

ostream& operator<<( ostream& os, const ZTriangle& object );

ZELOS_NAMESPACE_END

#endif

