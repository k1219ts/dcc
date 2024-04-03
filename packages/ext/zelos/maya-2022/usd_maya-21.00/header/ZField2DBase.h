//----------------//
// ZField2DBase.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.20                               //
//-------------------------------------------------------//

#ifndef _ZField2DBase_h_
#define _ZField2DBase_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZField2DBase : public ZGrid2D
{
	protected:

		int _numElements;
		int _iMax, _kMax;
		int _stride;
		ZFieldLocation::FieldLocation _location;

	public:

		ZField2DBase();
		ZField2DBase( const ZField2DBase& source );
		ZField2DBase( const ZGrid2D& grid, ZFieldLocation::FieldLocation loc );
		ZField2DBase( int Nx, int Nz, ZFieldLocation::FieldLocation loc );
		ZField2DBase( int Nx, int Nz, float Lx, float Lz, ZFieldLocation::FieldLocation loc );

		void set( const ZGrid2D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Nz, float Lx, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );

		void reset();

		ZField2DBase& operator=( const ZField2DBase& other );

		bool operator==( const ZField2DBase& other );
		bool operator!=( const ZField2DBase& other );

		bool directComputable( const ZField2DBase& other );
		bool directComputableWithoutLocation( const ZField2DBase& other );

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		int numElements() const { return _numElements; }
		int iMax() const { return _iMax; }
		int kMax() const { return _kMax; }
		ZFieldLocation::FieldLocation location() const { return _location; }

		int index( int i, int k ) const { return (i+_stride*k); }
		void index( const int n, int& i, int& k );

		int i0( int idx ) const { return ZMax( 0,            (idx-1)); }
		int i1( int idx ) const { return ZMin((idx+1),       _iMax  ); }
		int k0( int idx ) const { return ZMax((idx-_stride), 0      ); }
		int k1( int idx ) const { return ZMin((idx+_stride), _kMax  ); }

		ZPoint position( int i, int k ) const;
};

inline void
ZField2DBase::index( const int idx, int& i, int& k )
{
    i = idx%(_iMax+1);
    k = (idx/_stride)%(_kMax+1);
}

inline ZPoint
ZField2DBase::position( int i, int k ) const
{
	float x=i*_dx+_minPt.x, z=k*_dz+_minPt.z;
	if( _location==ZFieldLocation::zCell ) { x+=_dxd2; z+=_dxd2; }
	return ZPoint(x,_y,z);
}

ostream&
operator<<( ostream& os, const ZField2DBase& object );

////////////
// macros //

#define PER_EACH_ELEMENT_2D( field ) \
	for( int k=0; k<=field.kMax(); ++k ) { \
	for( int i=0; i<=field.iMax(); ++i ) {

#define PER_EACH_SUB_ELEMENT_2D( i0,i1,k0,k1 ) \
	for( int k=k0; k<=k1; ++k ) { \
	for( int i=i0; i<=i1; ++i ) {

#define PER_EACH_CELL_2D( field ) \
	for( int k=0; k<field.nz(); ++k ) { \
	for( int i=0; i<field.nx(); ++i ) {

#define PER_EACH_NODE_2D( field ) \
	for( int k=0; k<=field.nz(); ++k ) { \
	for( int i=0; i<=field.nx(); ++i ) {

#define END_PER_EACH_2D }}

ZELOS_NAMESPACE_END

#endif

