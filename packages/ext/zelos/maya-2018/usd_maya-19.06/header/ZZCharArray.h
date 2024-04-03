//---------------//
// ZZCharArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.16                               //
//-------------------------------------------------------//

#ifndef _ZZCharArray_h_
#define _ZZCharArray_h_

#include <ZelosCudaBase.h>

class ZZCharArray
{
	private:

		int   _n;		// element count
		char* _data;	// GPU memory pointer

	public:

		__host__
		ZZCharArray()
		: _n(0), _data(NULL)
		{}

		__host__
		ZZCharArray( const ZCharArray& a )
		: _n(0), _data(NULL)
		{
			from( a );
		}

		__host__
		~ZZCharArray()
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
			cudaError_t stat = cudaMalloc( (void**)&_data, _n*sizeof(char) );
			if( stat != cudaSuccess ) { cout<<"Error@ZZCharArray::setLength(): "<<cudaGetErrorString(stat)<<endl; return false; }
			return true;
		}

		__device__ __host__
		int length() const
		{
			return _n;
		}

		__device__
		char& operator[]( int i )
		{
			return _data[i];
		}

		__device__
		const char& operator[]( int i ) const
		{
			return _data[i];
		}

		__device__ __host__
		char* pointer()
		{
			return _data;
		}

		__device__ __host__
		const char* pointer() const
		{
			return _data;
		}

		__host__
		bool from( const ZCharArray& a )
		{
			setLength( a.length() );
			cudaError_t stat = cudaMemcpy( _data, &a[0], _n*sizeof(char), cudaMemcpyHostToDevice );
			if( stat != cudaSuccess ) { cout<<"Error@ZZCharArray::from(): "<<cudaGetErrorString(stat)<<endl; return false; }
			return true;
		}

		__host__
		bool to( ZCharArray& a )
		{
			a.setLength( _n );
			cudaError_t stat = cudaMemcpy( &a[0], _data, _n*sizeof(char), cudaMemcpyDeviceToHost );
			if( stat != cudaSuccess ) { cout<<"Error@ZZCharArray::to(): "<<cudaGetErrorString(stat)<<endl; return false; }
			return true;
		}
};

#endif

