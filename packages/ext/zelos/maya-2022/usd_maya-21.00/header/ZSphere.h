//-----------//
// ZSphere.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.29                               //
//-------------------------------------------------------//

#ifndef _ZSphere_h_
#define _ZSphere_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZSphere
{
	private:

		ZPoint _center;
		float  _radius;
		float  _r2;     // = _radius * _radius

	public:

		ZSphere();
		ZSphere( const ZSphere& sphere );
		ZSphere( const ZPoint& center, const float& radius );

		void reset();

		ZSphere& set( const ZPoint& center, const float& radius );
		ZSphere& set( const ZTriangle& triangle );

		ZSphere& operator=( const ZSphere& sphere );

		const ZPoint& center() const;
		const float& radius() const;
		const float& squaredRadius() const;

		bool contains( const ZPoint& point ) const;

		bool intersects( const ZLine& line ) const;
		bool intersects( const ZPoint& p0, const ZPoint& p1 ) const;

		void draw() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );
};

ostream& operator<<( ostream& os, const ZSphere& object );

ZELOS_NAMESPACE_END

#endif

