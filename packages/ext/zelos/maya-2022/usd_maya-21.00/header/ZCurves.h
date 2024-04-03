//-----------//
// ZCurves.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2019.03.07                               //
//-------------------------------------------------------//

#ifndef _ZCurves_h_
#define _ZCurves_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// Catmull-Rom spline based curve set class
// i: curve index
// j: CV index
// t: curve parameter
//
// cv(i,j) = position(t=j/(numCVs(i)-1))
class ZCurves
{
	private:

		// not to be saved
		ZIntArray    _numCVs;		// # of CVs                  (to be saved)
		ZIntArray    _startIdx;		// start index of each curve (not to be saved)
		ZPointArray  _cv;			// control vertex positions  (to be saved)

	public:

		ZCurves();
		ZCurves( const ZCurves& curve );
		ZCurves( const ZIntArray& nCVs );
		ZCurves( const char* filePathName );

		void reset();

		ZCurves& operator=( const ZCurves& others );

		void set( const ZIntArray& nCVs );

		void zeroize();

		void from( const vector<ZCurves>& curve, bool memAlloc );
		void from( const ZCurves& other, const ZCharArray& mask );

		void to( ZPointSetArray& cvs ) const;

		void append( const ZCurves& other );

		int numCurves() const;
		int numCVs( const int& i ) const;
		int numTotalCVs() const;
		int numSegments( const int& i ) const;

		const ZIntArray& numCVs() const;
		const ZIntArray& startIdx() const;
		const ZPointArray& cvs() const;
		ZPointArray& cvs();

		int globalIndex( const int& i, const int& j ) const;
		void getCVIndexRange( const int& i, int& start, int& end ) const;

		ZPoint& operator[]( const int& i );
		const ZPoint& operator[]( const int& i ) const;

		ZPoint& operator()( const int& i, const int& j );
		const ZPoint& operator()( const int& i, const int& j ) const;

		ZPoint& cv( const int& i, const int& j );
		const ZPoint& cv( const int& i, const int& j ) const;

		int rootIndex( const int& i ) const;
		int tipIndex( const int& i ) const;

		ZPoint& root( const int& i );
		const ZPoint& root( const int& i ) const;

		ZPoint& tip( const int& i );
		const ZPoint& tip( const int& i ) const;

		ZPoint  position ( int i, float t ) const;
		ZVector tangent  ( int i, float t ) const;
		ZVector normal   ( int i, float t ) const;
		ZVector biNormal ( int i, float t ) const;
		float   speed    ( int i, float t ) const;

		void getPositionAndTangent( int i, float t, ZPoint& P, ZVector& T ) const;

		float curveLength( int i ) const;
		float curveLength( int i, float t0, float t1 ) const;
		void getCurveLengths( ZFloatArray& curveLengths, bool useOpenMP=true ) const;

		float lineLength( int i ) const;
		void getLineLengths( ZFloatArray& curveLengths, bool useOpenMP=true ) const;

        ZBoundingBox boundingBox( int i ) const;

		ZBoundingBox boundingBox( bool onlyEndPoints=false, bool useOpenMP=true ) const;

		void getRootPoints( ZPointArray& rootPoints ) const;

		int deformByCCDIK( int i, const ZPoint& target, float rigidity, float maxAngleInDegrees, int maxIters, float tol );

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );

		double usedMemorySize( ZDataUnit::DataUnit dataUnit=ZDataUnit::zBytes ) const;

		void glPoints( int i, int j ) const;
		void glLine( int i, int j ) const;

		void drawLine( int i, float dt=-1.f ) const;
		void drawCVs( int i ) const;

		void drawLines( float dt ) const;
		void drawLines( float dt, ZColor color, float width, float colorRatio, float widthRatio ) const;
		void drawLines( float dt, ZCharArray& display, ZColorArray& color );
		void drawLines( float dt, ZCharArray& display, ZColorArray& color, float width, float colorRatio, float widthRatio ) const;

		void drawCVs() const;
		void drawCVs( ZCharArray& display, ZColorArray& color );

	private:

		void _init();
		void _allocate();

		void _whereIsIt( int i, float& t, int index[4] ) const;

		ZPoint  _zeroDerivative   ( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const;
		ZVector _firstDerivative  ( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const;
		ZVector _secondDerivative ( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const;
		ZVector _thirdDerivative  ( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const;
};

inline int
ZCurves::numCurves() const
{
	return (int)_numCVs.size();
}

inline int
ZCurves::numCVs( const int& i ) const
{
	return _numCVs[i];
}

inline int
ZCurves::numTotalCVs() const
{
	return _cv.length();
}

inline int
ZCurves::numSegments( const int& i ) const
{
	return (_numCVs[i]-1);
}

inline const ZIntArray&
ZCurves::numCVs() const
{
	return _numCVs;
}

inline const ZIntArray&
ZCurves::startIdx() const
{
	return _startIdx;
}

inline const ZPointArray&
ZCurves::cvs() const
{
	return _cv;
}

inline ZPointArray&
ZCurves::cvs()
{
	return _cv;
}

inline int
ZCurves::globalIndex( const int& i, const int& j ) const
{
	return ( _startIdx[i] + j );
}

inline void
ZCurves::getCVIndexRange( const int& i, int& start, int& end ) const
{
	end = ( start = _startIdx[i] ) + _numCVs[i] - 1;
}

inline ZPoint&
ZCurves::operator[]( const int& i )
{
	return _cv[i];
}

inline const
ZPoint& ZCurves::operator[]( const int& i ) const
{
	return _cv[i];
}

inline ZPoint&
ZCurves::operator()( const int& i, const int& j )
{
	return _cv[ _startIdx[i] + j ];
}

inline const
ZPoint& ZCurves::operator()( const int& i, const int& j ) const
{
	return _cv[ _startIdx[i] + j ];
}

inline ZPoint&
ZCurves::cv( const int& i, const int& j )
{
	return _cv[ _startIdx[i] + j ];
}

inline const
ZPoint& ZCurves::cv( const int& i, const int& j ) const
{
	return _cv[ _startIdx[i] + j ];
}

inline int
ZCurves::rootIndex( const int& i ) const
{
	return _startIdx[i];
}

inline int
ZCurves::tipIndex( const int& i ) const
{
	return ( _startIdx[i] + _numCVs[i] - 1 );
}

inline ZPoint&
ZCurves::root( const int& i )
{
	return _cv[ _startIdx[i] ];
}

inline const ZPoint&
ZCurves::root( const int& i ) const
{
	return _cv[ _startIdx[i] ];
}

inline ZPoint&
ZCurves::tip( const int& i )
{
	//return _cv[ _startIdx[i-1] - 1 ]; // it will fail for the last curve
	return _cv[ _startIdx[i] + _numCVs[i] - 1 ];
}

inline const ZPoint&
ZCurves::tip( const int& i ) const
{
	//return _cv[ _startIdx[i-1] - 1 ]; // it will fail for the last curve
	return _cv[ _startIdx[i] + _numCVs[i] - 1 ];
}

inline void
ZCurves::_whereIsIt( int i, float& t, int idx[4] ) const
{
	const int& nCVs   = _numCVs[i];
	const int  nCVs_1 = nCVs-1;

	t = ZClamp( t, 0.f, 1.f );

	const float k = t * nCVs_1;
	const int start = int(k);

	int& i0 = idx[0] = start-1;
	int& i1 = idx[1] = ( i0 >= nCVs_1 ) ? i0 : (i0+1);
	int& i2 = idx[2] = ( i1 >= nCVs_1 ) ? i1 : (i1+1);
	          idx[3] = ( i2 >= nCVs_1 ) ? i2 : (i2+1);

	if( i0 < 0 ) { i0 = 0; }

	t = k - start;
}

inline ZPoint
ZCurves::_zeroDerivative( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const
//	return 0.5f * ( (       2*P1           )
//				  + ( -1*P0     +1*P2      ) * t
//				  + (  2*P0-5*P1+4*P2-1*P3 ) * tt
//				  + ( -1*P0+3*P1-3*P2+1*P3 ) * ttt );
{
	const float tt  = t*t;
	const float ttt = tt*t;
	ZPoint tmp( WeightedSum( P0, P1, P2, P3, -t+2*tt-ttt, 2-5*tt+3*ttt, t+4*tt-3*ttt, -tt+ttt ) );
	return ( tmp*=0.5f );
}

inline ZVector
ZCurves::_firstDerivative( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const
//	return 0.5f * (   ( -1*P0     +1*P2      )
//				  + 2*(  2*P0-5*P1+4*P2-1*P3 )*t
//				  + 3*( -1*P0+3*P1-3*P2+1*P3 )*tt );
{
	const float tt = t*t;
	ZVector tmp( WeightedSum( P0, P1, P2, P3, -1+4*t-3*tt, -10*t+9*tt, 1+8*t-9*tt, -2*t+3*tt ) );
	return ( tmp*=0.5f );
}

inline ZVector
ZCurves::_secondDerivative( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const
//	return 0.5f * ( 2*(  2*P0-5*P1+4*P2-1*P3 )
//				  + 6*( -1*P0+3*P1-3*P2+1*P3 )*t );
{
	ZVector tmp( WeightedSum( P0, P1, P2, P3, 4-6*t, -10+18*t, 8-18*t, -2+6*t ) );
	return ( tmp*=0.5f );
}

inline ZVector
ZCurves::_thirdDerivative( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const
//	return 0.5f * ( 6*( -1*P0+3*P1-3*P2+1*P3 ) );
{
	ZVector tmp( WeightedSum( P0, P1, P2, P3, -6.f, 18.f, -18.f, 6.f ) );
	return ( tmp*=0.5f );
}

inline void
ZCurves::glPoints( int i, int j ) const
{
	glVertex( cv(i,j) );
}

inline void
ZCurves::glLine( int i, int j ) const
{
	glVertex( cv(i,j  ) );
	glVertex( cv(i,j+1) );
}

ostream&
operator<<( ostream& os, const ZCurves& object );

ZELOS_NAMESPACE_END

#endif

