//--------------//
// ZZIntArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.16                               //
//-------------------------------------------------------//

#ifndef _ZZIntArray_h_
#define _ZZIntArray_h_

#include <ZelosCudaBase.h>

class ZZIntArray
{
	private:

		int  _n;	// element count
		int* _data;	// GPU memory pointer

	public:

		__host__
		ZZIntArray()
		: _n(0), _data(NULL)
		{}

		__host__
		ZZIntArray( const ZIntArray& a )
		: _n(0), _data(NULL)
		{
			from( a );
		}

		__host__
		~ZZIntArray()
		{
			reset();
		} 

		__host__
		void reset()
		{
			if( _data ) { cudaFree( _data ); }
			_n = 0;
		}

		__host__
		bool setLength( int l )
		{
			if( _n == l ) { return true; }
			reset();
			_n = l;
			cudaError_t stat = cudaMalloc( (void**)&_data, _n*sizeof(int) );
			if( stat != cudaSuccess ) { cout<<"Error@ZZIntArray::setLength(): "<<cudaGetErrorString(stat)<<endl; return false; }
			return true;
		}

		__device__ __host__
		int length() const
		{
			return _n;
		}

		__device__
		int& operator[]( int i )
		{
			return _data[i];
		}

		__device__
		const int& operator[]( int i ) const
		{
			return _data[i];
		}

		__device__ __host__
		int* pointer()
		{
			return _data;
		}

		__device__ __host__
		const int* pointer() const
		{
			return _data;
		}

		__host__
		bool from( const ZIntArray& a )
		{
			setLength( a.length() );
			cudaError_t stat = cudaMemcpy( _data, &a[0], _n*sizeof(int), cudaMemcpyHostToDevice );
			if( stat != cudaSuccess ) { cout<<"Error@ZZIntArray::from(): "<<cudaGetErrorString(stat)<<endl; return false; }
			return true;
		}

		__host__
		bool to( ZIntArray& a )
		{
			a.setLength( _n );
			cudaError_t stat = cudaMemcpy( &a[0], _data, _n*sizeof(int), cudaMemcpyDeviceToHost );
			if( stat != cudaSuccess ) { cout<<"Error@ZZIntArray::to(): "<<cudaGetErrorString(stat)<<endl; return false; }
			return true;
		}
};

#endif

