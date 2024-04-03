//---------//
// ZLine.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.29                               //
//-------------------------------------------------------//

#ifndef _ZLine_h_
#define _ZLine_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZLine
{
	private:

		ZPoint _a;
		ZPoint _b;

	public:

		ZLine();
		ZLine( const ZLine& line );
		ZLine( const ZPoint& a, const ZPoint& b );

		void reset();

		ZLine& set( const ZPoint& a, const ZPoint& b );

		ZLine& operator=( const ZLine& line );

		const ZPoint& a() const;
		const ZPoint& b() const;

		ZPoint center() const;

		float length() const;

		void draw() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );
};

ostream& operator<<( ostream& os, const ZLine& object );

ZELOS_NAMESPACE_END

#endif

