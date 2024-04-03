//-----------------//
// ZZIntSetArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.16                               //
//-------------------------------------------------------//

#ifndef _ZZIntSetArray_h_
#define _ZZIntSetArray_h_

#include <ZelosCudaBase.h>

class ZZIntSetArray
{
	private:

		int  _nl;
		int  _il;
		int  _vl;

		int* _n;
		int* _i;
		int* _v;

	public:

		__host__
		ZZIntSetArray()
		: _n(NULL), _i(NULL), _v(NULL), _nl(0), _il(0), _vl(0)
		{}

		__host__
		ZZIntSetArray( const ZIntSetArray& a )
		: _n(NULL), _i(NULL), _v(NULL), _nl(0), _il(0), _vl(0)
		{
			set( a );
		}

		__host__
		ZZIntSetArray( const vector<ZIntSetArray>& a )
		: _n(NULL), _i(NULL), _v(NULL), _nl(0), _il(0), _vl(0)
		{
			set( a );
		}

		__host__
		~ZZIntSetArray()
		{
			reset();
		} 

		__host__
		void reset()
		{
			cudaFree( _n );
			cudaFree( _i );
			cudaFree( _v );

			_n = (int*)NULL;
			_i = (int*)NULL;
			_v = (int*)NULL;

			_nl = 0;
			_il = 0;
			_vl = 0;
		}

		__host__
		bool set( const ZIntSetArray& a )
		{
			const ZIntArray& n = a.n();
			const ZIntArray& i = a.i();
			const ZIntArray& v = a.v();

			const int nl = n.length();
			const int il = i.length();
			const int vl = v.length();

			if( nl != _nl )
			{
				cudaFree( _n );
				cudaError_t stat = cudaMalloc( (void**)&_n, nl*sizeof(int) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZIntSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_nl = nl;
			}

			if( il != _il )
			{
				cudaFree( _i );
				cudaError_t stat = cudaMalloc( (void**)&_i, il*sizeof(int) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZIntSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_il = il;
			}

			if( vl != _vl )
			{
				cudaFree( _v );
				cudaError_t stat = cudaMalloc( (void**)&_v, vl*sizeof(int) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZIntSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_vl = vl;
			}

			cudaMemcpy( _n, &n[0], nl*sizeof(int), cudaMemcpyHostToDevice );
			cudaMemcpy( _i, &i[0], il*sizeof(int), cudaMemcpyHostToDevice );
			cudaMemcpy( _v, &v[0], vl*sizeof(int), cudaMemcpyHostToDevice );

			return true;
		}

		__host__
		bool set( const vector<ZIntSetArray>& a )
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
				if( stat != cudaSuccess ) { cout<<"Error@ZZIntSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_nl = nl;
			}

			if( il != _il )
			{
				cudaFree( _i );
				cudaError_t stat = cudaMalloc( (void**)&_i, il*sizeof(int) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZIntSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_il = il;
			}

			if( vl != _vl )
			{
				cudaFree( _v );
				cudaError_t stat = cudaMalloc( (void**)&_v, vl*sizeof(int) );
				if( stat != cudaSuccess ) { cout<<"Error@ZZIntSetArray::set(): "<<cudaGetErrorString(stat)<<endl; return false; }
				_vl = vl;
			}

			int nCount = 0;
			int iCount = 0;
			int vCount = 0;

			FOR( g, 0, nGrp )
			{
				const ZIntArray& n = a[g].n();
				nl = n.length();
				cudaMemcpy( &_n[nCount], &n[0], nl*sizeof(int), cudaMemcpyHostToDevice );
				nCount += nl;

				ZIntArray i( a[g].i() );
				il = i.length();
				FOR( j, 0, il ) { i[j] += iCount; }
				cudaMemcpy( &_i[iCount], &i[0], il*sizeof(int), cudaMemcpyHostToDevice );
				iCount += il;

				const ZIntArray& v = a[g].v();
				vl = v.length();
				cudaMemcpy( &_v[vCount], &v[0], vl*sizeof(int), cudaMemcpyHostToDevice );
				vCount += vl;
			}

			return true;
		}

		__device__
		int& operator()( int i, int j )
		{
			return _v[ _i[i] + j ];
		}

		__device__
		const int& operator()( int i, int j ) const
		{
			return _v[ _i[i] + j ];
		}

		__device__
		int& start( int i )
		{
			return _v[ _i[i] ];
		}

		__device__
		const int& start( int i ) const
		{
			return _v[ _i[i] ];
		}

		__device__
		int& end( int i )
		{
			return _v[ _i[i] + _n[i] - 1 ];
		}

		__device__
		const int& end( int i ) const
		{
			return _v[ _i[i] + _n[i] - 1 ];
		}
};

#endif

