//----------------//
// ZZFloatArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.16                               //
//-------------------------------------------------------//

#ifndef _ZZFloatArray_h_
#define _ZZFloatArray_h_

#include <ZelosCudaBase.h>

class ZZFloatArray
{
	private:

		int    _n;		// element count
		float* _data;	// GPU memory pointer

	public:

		__host__
		ZZFloatArray()
		: _n(0), _data(NULL)
		{}

		__host__
		ZZFloatArray( const ZFloatArray& a )
		: _n(0), _data(NULL)
		{
			from( a );
		}

		__host__
		ZZFloatArray( const ZVectorArray& a )
		: _n(0), _data(NULL)
		{
			from( a );
		}

		__host__
		~ZZFloatArray()
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
			cudaError_t stat = cudaMalloc( (void**)&_data, _n*sizeof(float) );
			if( stat != cudaSuccess ) { cout<<"Error@ZZFloatArray::setLength(): "<<cudaGetErrorString(stat)<<endl; return false; }
			return true;
		}

		__device__ __host__
		int length() const
		{
			return _n;
		}

		__device__
		float& operator[]( int i )
		{
			return _data[i];
		}

		__device__
		const float& operator[]( int i ) const
		{
			return _data[i];
		}

		__device__ __host__
		float* pointer()
		{
			return _data;
		}

		__device__ __host__
		const float* pointer() const
		{
			return _data;
		}

		__host__
		bool from( const ZFloatArray& a )
		{
			setLength( a.length() );
			cudaError_t stat = cudaMemcpy( _data, &a[0], _n*sizeof(float), cudaMemcpyHostToDevice );
			if( stat != cudaSuccess ) { cout<<"Error@ZZFloatArray::from(): "<<cudaGetErrorString(stat)<<endl; return false; }

			return true;
		}

		__host__
		bool from( const ZVectorArray& a )
		{
			setLength( 3*a.length() );
			cudaError_t stat = cudaMemcpy( (void*)_data, (const void*)&a[0].x, 3*_n*sizeof(float), cudaMemcpyHostToDevice );
			if( stat != cudaSuccess ) { cout<<"Error@ZZFloatArray::from(): "<<cudaGetErrorString(stat)<<endl; return false; }
			return true;
		}

		__host__
		bool to( ZFloatArray& a )
		{
			a.setLength( _n );
			cudaError_t stat = cudaMemcpy( &a[0], _data, _n*sizeof(float), cudaMemcpyDeviceToHost );
			if( stat != cudaSuccess ) { cout<<"Error@ZZFloatArray::to(): "<<cudaGetErrorString(stat)<<endl; return false; }
			return true;
		}

		__host__
		bool to( ZVectorArray& a )
		{
			a.setLength( _n );
			cudaError_t stat = cudaMemcpy( &a[0].x, _data, 3*_n*sizeof(float), cudaMemcpyDeviceToHost );
			if( stat != cudaSuccess ) { cout<<"Error@ZZFloatArray::to(): "<<cudaGetErrorString(stat)<<endl; return false; }
			return true;
		}
};

#endif

