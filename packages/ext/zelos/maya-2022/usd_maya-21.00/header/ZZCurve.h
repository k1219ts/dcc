//------------//
// ZZCurves.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.07.15                               //
//-------------------------------------------------------//

#ifndef _ZZCurves_h_
#define _ZZCurves_h_

#include <ZelosCudaBase.h>

class ZZCurves
{
	//private:
	public:

		int      _nCurves;
		int      _nTotalCVs;

		int*     _numCVs;
		int*     _startIdx;
		ZZPoint* _cv;

	public:

		__host__
		ZZCurves()
		{
			_nCurves   = 0;
			_nTotalCVs = 0;

			_numCVs    = (int* )NULL;
			_startIdx  = (int* )NULL;
			_cv        = (ZZPoint*)NULL;
		}

		__host__
		ZZCurves( const ZCurves& cpuCurves )
		{
			from( cpuCurves );
		}

		__host__
		~ZZCurves()
		{
			reset();
		}

		__host__
		void reset()
		{
			if( _numCVs   ) { cudaFree( _numCVs   ); }
			if( _startIdx ) { cudaFree( _startIdx ); }
			if( _cv       ) { cudaFree( _cv       ); }
		}

		__host__
		void from( const ZCurves& cpuCurves )
		{
			_nCurves   = cpuCurves.numCurves();
			_nTotalCVs = cpuCurves.numTotalCVs();

			const ZIntArray& numCVs = cpuCurves.numCVs();
			cudaMalloc( (void**)&_numCVs, _nCurves*sizeof(int) );
			cudaMemcpy( _numCVs, &numCVs[0], _nCurves*sizeof(int), cudaMemcpyHostToDevice );

			const ZIntArray& startIdx = cpuCurves.startIdx();
			cudaMalloc( (void**)&_startIdx, _nCurves*sizeof(int) );
			cudaMemcpy( _startIdx, &startIdx[0], _nCurves*sizeof(int), cudaMemcpyHostToDevice );

			const ZPointArray& cv = cpuCurves.cvs();
			cudaMalloc( (void**)&_cv, _nTotalCVs*sizeof(float*)*3 );
			cudaMemcpy( _cv, &cv[0].x, _nTotalCVs*sizeof(ZPoint), cudaMemcpyHostToDevice );
		}

		__host__
		void to( ZCurves& cpuCurves )
		{
			if( !_nTotalCVs )
			{
				cpuCurves.reset();
				return;
			}

			ZPointArray& cv = cpuCurves.cvs();
			cv.setLength( _nTotalCVs );

			cudaMemcpy( &cv[0].x, _cv, _nTotalCVs*sizeof(ZPoint), cudaMemcpyDeviceToHost );
		}

		__device__
		int numCurves() const
		{
			return _nCurves;
		}

		__device__
		int numCVs( int i ) const
		{
			return _numCVs[i];
		}

		__device__
		int numTotalCVs() const
		{
			return _nTotalCVs;
		}

		__device__
		ZZPoint& cv( int i, int j )
		{
			return _cv[ _startIdx[i] + j ];
		}

		__device__
		const ZZPoint& cv( int i, int j ) const
		{
			return _cv[ _startIdx[i] + j ];
		}

		__device__
		ZZPoint& root( int i )
		{
			return _cv[ _startIdx[i] ];
		}

		__device__
		const ZZPoint& root( int i ) const
		{
			return _cv[ _startIdx[i] ];
		}

		__device__
		ZZPoint& tip( int i )
		{
			return _cv[ _startIdx[i] + _numCVs[i] - 1 ];
		}

		__device__
		const ZZPoint& tip( int i ) const
		{
			return _cv[ _startIdx[i] + _numCVs[i] - 1 ];
		}

		__device__
		ZZPoint position( int i, float t ) const
		{
			int idx[4];
			_whereIsIt( i, t, idx );

			const ZZPoint& P0 = cv( i, idx[0] );
			const ZZPoint& P1 = cv( i, idx[1] );
			const ZZPoint& P2 = cv( i, idx[2] );
			const ZZPoint& P3 = cv( i, idx[3] );

			return _zeroDerivative( t, P0, P1, P2, P3 );
		}

	private:

		__device__
		void _whereIsIt( int i, float& t, int idx[4] ) const
		{
			const int& nCVs   = _numCVs[i];
			const int  nCVs_1 = nCVs-1;

			t = ZZClamp( t, 0.f, 1.f );

			const float k = t * nCVs_1;
			const int start = int(k);

			int& i0 = idx[0] = start-1;
			int& i1 = idx[1] = ( i0 >= nCVs_1 ) ? i0 : (i0+1);
			int& i2 = idx[2] = ( i1 >= nCVs_1 ) ? i1 : (i1+1);
			int& i3 = idx[3] = ( i2 >= nCVs_1 ) ? i2 : (i2+1);

			if( i0 < 0 ) { i0 = 0; }

			t = k - start;
		}

		__device__
		ZZPoint _zeroDerivative( float t, const ZZPoint& P0, const ZZPoint& P1, const ZZPoint& P2, const ZZPoint& P3 ) const
		{
			ZZPoint p;

			float w0 = 0.5f * (         2*P1.x               );
			float w1 = 0.5f * ( -1*P0.x       +1*P2.x        ) * t;
			float w2 = 0.5f * (  2*P0.x-5*P1.x+4*P2.x-1*P3.x ) * (t*t);
			float w3 = 0.5f * ( -1*P0.x+3*P1.x-3*P2.x+1*P3.x ) * (t*t*t);
			p.x = w0 * P0.x + w1 * P1.x + w2 * P2.x + w3 * P3.x;

			w0 = 0.5f * (         2*P1.y               );
			w1 = 0.5f * ( -1*P0.y       +1*P2.y        ) * t;
			w2 = 0.5f * (  2*P0.y-5*P1.y+4*P2.y-1*P3.y ) * (t*t);
			w3 = 0.5f * ( -1*P0.y+3*P1.y-3*P2.y+1*P3.y ) * (t*t*t);
			p.y = w0 * P0.y + w1 * P1.y + w2 * P2.y + w3 * P3.y;

			w0 = 0.5f * (         2*P1.z               );
			w1 = 0.5f * ( -1*P0.z       +1*P2.z        ) * t;
			w2 = 0.5f * (  2*P0.z-5*P1.z+4*P2.z-1*P3.z ) * (t*t);
			w3 = 0.5f * ( -1*P0.z+3*P1.z-3*P2.z+1*P3.z ) * (t*t*t);
			p.z = w0 * P0.z + w1 * P1.z + w2 * P2.z + w3 * P3.z;

			return p;
		}
};

#endif

