//----------------//
// ZField3DBase.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZField3DBase_h_
#define _ZField3DBase_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZField3DBase : public ZGrid3D
{
	protected:

		int _numElements;
		int _iMax, _jMax, _kMax;
		int _stride0, _stride1;
		ZFieldLocation::FieldLocation _location;

	public:

		ZField3DBase();
		ZField3DBase( const ZField3DBase& source );
		ZField3DBase( const ZGrid3D& grid, ZFieldLocation::FieldLocation loc );
		ZField3DBase( int Nx, int Ny, int Nz, ZFieldLocation::FieldLocation loc );
		ZField3DBase( int Nx, int Ny, int Nz, float Lx, float Ly, float Lz, ZFieldLocation::FieldLocation loc );

		void set( const ZGrid3D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Ny, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Ny, int Nz, float Lx, float Ly, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );

		void reset();

		ZField3DBase& operator=( const ZField3DBase& other );

		bool operator==( const ZField3DBase& other );
		bool operator!=( const ZField3DBase& other );

		bool directComputable( const ZField3DBase& other );
		bool directComputableWithoutLocation( const ZField3DBase& other );

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		int numElements() const { return _numElements; }
		int iMax() const { return _iMax; }
		int jMax() const { return _jMax; }
		int kMax() const { return _kMax; }
		ZFieldLocation::FieldLocation location() const { return _location; }

		int index( int i, int j, int k ) const { return (i+_stride0*j+_stride1*k); }
        void index( const int n, int& i, int& j, int& k ) const;

		int i0( int idx ) const { return (idx-1);        }
		int i1( int idx ) const { return (idx+1);        }
		int j0( int idx ) const { return (idx-_stride0); }
		int j1( int idx ) const { return (idx+_stride0); }
		int k0( int idx ) const { return (idx-_stride1); }
		int k1( int idx ) const { return (idx+_stride1); }

		ZPoint position( int i, int j, int k ) const;

		void getLerpIndicesAndWeights( const ZPoint& p, int* indices, float* weights ) const;

		// 27 node indices of (i,j,k) cell
		void getNeighnorCells( int i, int j, int k, int indices[27] )
		{
			i = ZClamp( i, 1, _iMax-1 );
			j = ZClamp( j, 1, _jMax-1 );
			k = ZClamp( k, 1, _kMax-1 );
			const int idx = index( i,j,k );

			const int jj = _stride0;
			const int kk = _stride1;

			indices[0] = idx-1-jj-kk; // (i-1, j-1, k-1 );
			indices[1] = idx-jj-kk;   // (i  , j-1, k-1 );
			indices[2] = idx+1-jj-kk; // (i+1, j-1, k-1 );
			indices[3] = idx-1-kk;    // (i-1, j  , k-1 );
			indices[4] = idx-kk;  	  // (i  , j  , k-1 );
			indices[5] = idx+1-kk;	  // (i+1, j  , k-1 );
			indices[6] = idx-1+jj-kk; // (i-1, j+1, k-1 );
			indices[7] = idx+jj-kk;   // (i  , j+1, k-1 );
			indices[8] = idx+1+jj-kk; // (i+1, j+1, k-1 );

			indices[9]  = idx-1-jj;   // (i-1, j-1, k );
			indices[10] = idx-jj;     // (i  , j-1, k );
			indices[11] = idx+1-jj;   // (i+1, j-1, k );
			indices[12] = idx-1;      // (i-1, j  , k );
			indices[13] = idx;  	  // (i  , j  , k );
			indices[14] = idx+1;	  // (i+1, j  , k );
			indices[15] = idx-1+jj;   // (i-1, j+1, k );
			indices[16] = idx+jj;     // (i  , j+1, k );
			indices[17] = idx+1+jj;   // (i+1, j+1, k );

			indices[18] = idx-1-jj+kk;// (i-1, j-1, k+1 );
			indices[19] = idx-jj+kk;  // (i  , j-1, k+1 );
			indices[20] = idx+1-jj+kk;// (i+1, j-1, k+1 );
			indices[21] = idx-1+kk;   // (i-1, j  , k+1 );
			indices[22] = idx+kk;  	  // (i  , j  , k+1 );
			indices[23] = idx+1+kk;	  // (i+1, j  , k+1 );
			indices[24] = idx-1+jj+kk;// (i-1, j+1, k+1 );
			indices[25] = idx+jj+kk;  // (i  , j+1, k+1 );
			indices[26] = idx+1+jj+kk;// (i+1, j+1, k+1 );
		}
		
};

inline void 
ZField3DBase::index(const int idx, int& i, int& j, int& k) const
{
    i = idx%(_iMax+1);
    j = (idx/_stride0)%(_jMax+1);
    k = (idx/_stride1)%(_kMax+1);
}

inline ZPoint
ZField3DBase::position( int i, int j, int k ) const
{
	float x=i*_dx+_minPt.x, y=j*_dy+_minPt.y, z=k*_dz+_minPt.z;
	if( _location==ZFieldLocation::zCell ) { x+=_dxd2; y+=_dyd2; z+=_dxd2; }
	return ZPoint(x,y,z);
}

inline void
ZField3DBase::getLerpIndicesAndWeights( const ZPoint& p, int* indices, float* weights ) const
{
    float x=p.x, y=p.y, z=p.z;
	if( _location==ZFieldLocation::zCell ){ x-=_dxd2; y-=_dyd2; z-=_dzd2; }

	x = std::max(std::min(x, _maxPt.x), _minPt.x);
	y = std::max(std::min(y, _maxPt.y), _minPt.y);
	z = std::max(std::min(z, _maxPt.z), _minPt.z);

	int i=int(x=((x-_minPt.x)*_ddx)); float fx=x-i;
	int j=int(y=((y-_minPt.y)*_ddy)); float fy=y-j;
	int k=int(z=((z-_minPt.z)*_ddz)); float fz=z-k;

	if(i<0) {i=0;fx=0;} else if(i>=_iMax) {i=_iMax-1;fx=1;}
	if(j<0) {j=0;fy=0;} else if(j>=_jMax) {j=_jMax-1;fy=1;}
	if(k<0) {k=0;fz=0;} else if(k>=_kMax) {k=_kMax-1;fz=1;}

	int idx[8];
	idx[0]=i+_stride0*j+_stride1*k; idx[1]=idx[0]+1; idx[2]=idx[1]+_stride1; idx[3]=idx[2]-1;
	idx[4]=idx[0]+_stride0;         idx[5]=idx[4]+1; idx[6]=idx[5]+_stride1; idx[7]=idx[6]-1;

	const float _fx=1-fx, _fy=1-fy, _fz=1-fz;
	const float wgt[8] = { _fx*_fy*_fz, fx*_fy*_fz, fx*_fy*fz, _fx*_fy*fz, _fx*fy*_fz, fx*fy*_fz, fx*fy*fz, _fx*fy*fz };

	FOR(l,0,8) { indices[l] = idx[l]; weights[l] = wgt[l]; }
}

ostream&
operator<<( ostream& os, const ZField3DBase& object );

////////////
// macros //

#define PER_EACH_ELEMENT_3D( field ) \
	for( int k=0; k<=field.kMax(); ++k ) { \
	for( int j=0; j<=field.jMax(); ++j ) { \
	for( int i=0; i<=field.iMax(); ++i ) {

#define PER_EACH_ELEMENT_WITHOUT_BORDER_3D( field ) \
	for( int k=1; k<field.kMax(); ++k ) { \
	for( int j=1; j<field.jMax(); ++j ) { \
	for( int i=1; i<field.iMax(); ++i ) {

#define PER_EACH_SUB_ELEMENT_3D( i0,i1,j0,j1,k0,k1 ) \
	for( int k=k0; k<=k1; ++k ) { \
	for( int j=j0; j<=j1; ++j ) { \
	for( int i=i0; i<=i1; ++i ) {

#define PER_EACH_CELL_3D( field ) \
	for( int k=0; k<field.nz(); ++k ) { \
	for( int j=0; j<field.ny(); ++j ) { \
	for( int i=0; i<field.nx(); ++i ) {

#define PER_EACH_NODE_3D( field ) \
	for( int k=0; k<=field.nz(); ++k ) { \
	for( int j=0; j<=field.ny(); ++j ) { \
	for( int i=0; i<=field.nx(); ++i ) {

#define END_PER_EACH_3D }}}

ZELOS_NAMESPACE_END

#endif

