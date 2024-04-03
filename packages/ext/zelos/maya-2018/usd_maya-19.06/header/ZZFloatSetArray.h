//-------------------//
// ZZFloatSetArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.16                               //
//-------------------------------------------------------//

#ifndef _ZZFloatSetArray_h_
#define _ZZFloatSetArray_h_

#include <ZelosCudaBase.h>

class ZZFloatSetArray
{
	private:

		int    _nl;
		int    _il;
		int    _vl;

		int*   _n;
		int*   _i;
		float* _v;

	public:

		__host__
		ZZFloatSetArray()
		: _n(NULL), _i(NULL), _v(NULL), _nl(0), _il(0), _vl(0)
		{}

		__host__
		ZZFloatSetArray( const ZFloatSetArray& a )
		: _n(NULL), _i(NULL), _v(NULL), _nl(0), _il(0), _vl(0)
		{
			set( a );
		}

		__host__
		ZZFloatSetArray( const vector<ZFloatSetArray>& a )
		: _n(NULL), _i(NULL), _v(NULL), _nl(0), _il(0), _vl(0)
		{
			set( a );
		}

		__host__
		~ZZFloatSetArray()
		{
			reset();
		} 

		__host__
		void reset()
		{
			cudaFree( _n );
			cudaFree( _i );
			cudaFree( _v );

			_n = (int*  )NULL;
			_i = (int*  )NULL;
			_v = (float*)NULL;

			_nl = 0;
			_il = 0;
			_vl = 0;
		}

		__host__
		bool set( const ZFloatSetArray& a )
		{
			const ZIntArray&   n = a.n();
			const ZIntArray&   i = a.i();
			const ZFloatArray& v = a.v();

			const int nl = n.length();
			const int il = i.length();
			const int vl = v.length();

			if( nl != _nl )
			{
				cudaFree( _n );
				cudaError_t stat = cudaMalloc( (void**)&_n, nl*sizeof(int) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZFloatSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_nl = nl;
			}

			if( il != _il )
			{
				cudaFree( _i );
				cudaError_t stat = cudaMalloc( (void**)&_i, il*sizeof(int) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZFloatSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_il = il;
			}

			if( vl != _vl )
			{
				cudaFree( _v );
				cudaError_t stat = cudaMalloc( (void**)&_v, vl*sizeof(float) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZFloatSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_vl = vl;
			}

			cudaMemcpy( _n, &n[0], nl*sizeof(int),   cudaMemcpyHostToDevice );
			cudaMemcpy( _i, &i[0], il*sizeof(int),   cudaMemcpyHostToDevice );
			cudaMemcpy( _v, &v[0], vl*sizeof(float), cudaMemcpyHostToDevice );

			return true;
		}

		__host__
		bool set( const vector<ZFloatSetArray>& a )
		{
			const int nGrp = (int)a.size();
			if( !nGrp ) { reset(); return true; }

			int nl = 0;
			int il = 0;
			int vl = 0;

			FOR( g, 0, nGrp )
			{
				nl += a[g].n().length();
				il += a[g].i().length();
				vl += a[g].v().length();
			}

			if( nl != _nl )
			{
				cudaFree( _n );
				cudaError_t stat = cudaMalloc( (void**)&_n, nl*sizeof(int) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZFloatSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_nl = nl;
			}

			if( il != _il )
			{
				cudaFree( _i );
				cudaError_t stat = cudaMalloc( (void**)&_i, il*sizeof(int) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZFloatSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_il = il;
			}

			if( vl != _vl )
			{
				cudaFree( _v );
				cudaError_t stat = cudaMalloc( (void**)&_v, vl*sizeof(float) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZFloatSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_vl = vl;
			}

			int nCount = 0;
			int iCount = 0;
			int vCount = 0;

			FOR( g, 0, nGrp )
			{
				const ZIntArray& n = a[g].n();
				nl = n.length();
				cudaMemcpy( &_n[nCount], &n[0], nl*sizeof(int),   cudaMemcpyHostToDevice );
				nCount += nl;

				ZIntArray i( a[g].i() );
				il = i.length();
				FOR( j, 0, il ) { i[j] += iCount; }
				cudaMemcpy( &_i[iCount], &i[0], il*sizeof(int),   cudaMemcpyHostToDevice );
				iCount += il;

				const ZFloatArray& v = a[g].v();
				vl = v.length();
				cudaMemcpy( &_v[vCount], &v[0], vl*sizeof(float), cudaMemcpyHostToDevice );
				vCount += vl;
			}

			return true;
		}

		__device__
		float& operator()( int i, int j )
		{
			return _v[ _i[i] + j ];
		}

		__device__
		const float& operator()( int i, int j ) const
		{
			return _v[ _i[i] + j ];
		}

		__device__
		float& start( int i )
		{
			return _v[ _i[i] ];
		}

		__device__
		const float& start( int i ) const
		{
			return _v[ _i[i] ];
		}

		__device__
		float& end( int i )
		{
			return _v[ _i[i] + _n[i] - 1 ];
		}

		__device__
		const float& end( int i ) const
		{
			return _v[ _i[i] + _n[i] - 1 ];
		}
};

#endif

