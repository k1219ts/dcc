//------------------//
// ZMarkerField2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.05                               //
//-------------------------------------------------------//

#ifndef _ZMarkerField2D_h_
#define _ZMarkerField2D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZMarkerField2D : public ZField2DBase, public ZIntArray
{
	public:

		ZMarkerField2D();
		ZMarkerField2D( const ZMarkerField2D& source );
		ZMarkerField2D( const ZGrid2D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZMarkerField2D( int Nx, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZMarkerField2D( int Nx, int Nz, float Lx, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZMarkerField2D( const char* filePathName );

		void set( const ZGrid2D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Nz, float Lx, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );

		void reset();

		ZMarkerField2D& operator=( const ZMarkerField2D& other );

		bool exchange( const ZMarkerField2D& other );

		int& operator()( const int& i, const int& k );
		const int& operator()( const int& i, const int& k ) const;

		const ZString dataType() const;

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

inline int&
ZMarkerField2D::operator()( const int& i, const int& k )
{
	return std::vector<int>::operator[]( i + _stride*k );
}

inline const int&
ZMarkerField2D::operator()( const int& i, const int& k ) const
{
	return std::vector<int>::operator[]( i + _stride*k );
}

ostream&
operator<<( ostream& os, const ZMarkerField2D& object );

ZELOS_NAMESPACE_END

#endif

