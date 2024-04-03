//------------------//
// ZMarkerField3D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.05                               //
//-------------------------------------------------------//

#ifndef _ZMarkerField3D_h_
#define _ZMarkerField3D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZMarkerField3D : public ZField3DBase, public ZIntArray
{
	public:

		ZMarkerField3D();
		ZMarkerField3D( const ZMarkerField3D& source );
		ZMarkerField3D( const ZGrid3D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZMarkerField3D( int Nx, int Ny, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZMarkerField3D( int Nx, int Ny, int Nz, float Lx, float Ly, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZMarkerField3D( const char* filePathName );

		void set( const ZGrid3D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Ny, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Ny, int Nz, float Lx, float Ly, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );

		void reset();

		ZMarkerField3D& operator=( const ZMarkerField3D& other );

		bool exchange( const ZMarkerField3D& other );

		int& operator()( const int& i, const int& j, const int& k );
		const int& operator()( const int& i, const int& j, const int& k ) const;

		const ZString dataType() const;

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

inline int&
ZMarkerField3D::operator()( const int& i, const int& j, const int& k )
{
	return std::vector<int>::operator[]( i + _stride0*j + _stride1*k );
}

inline const int&
ZMarkerField3D::operator()( const int& i, const int& j, const int& k ) const
{
	return std::vector<int>::operator[]( i + _stride0*j + _stride1*k );
}

ostream&
operator<<( ostream& os, const ZMarkerField3D& object );

ZELOS_NAMESPACE_END

#endif

