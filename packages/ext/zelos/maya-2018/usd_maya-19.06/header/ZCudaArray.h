//--------------//
// ZCudaArray.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.15                               //
//-------------------------------------------------------//

#ifndef _ZCudaArray_h_
#define _ZCudaArray_h_

#include <ZelosCudaBase.h>

template <typename T>
class ZCudaArray 
{
	private:

		bool _pinned;
		int  _size;
		T*   _hostPtr;
		T*   _devicePtr;

	public:

		enum Direction { HOST_TO_DEVICE, DEVICE_TO_HOST };

	public:

		ZCudaArray();
		ZCudaArray( int n, bool pinnedMemory=false );

		~ZCudaArray();

		void free();

		bool setLength( int n, bool pinnedMemory=false );
		bool copy( Direction dir );

		T* hostPtr();
		T* devicePtr();

		int length() const;

	private:

		void _init();

		bool _copyHostToDevice();
		bool _copyDeviceToHost();

		bool _allocHost( bool pinnedMemory );
		bool _allocDevice();

		void _freeHost();
		void _freeDevice();
};

template <typename T> 
ZCudaArray<T>::ZCudaArray()
{
	_init();
}

template <typename T>
ZCudaArray<T>::ZCudaArray( int n, bool pinned )
{
	_init();
	setLength( n, pinned );
}

template <typename T>
void
ZCudaArray<T>::_init()
{
	_pinned    = false;
	_size      = 0;
	_hostPtr   = (T*)NULL;
	_devicePtr = (T*)NULL;
}

template <typename T>
ZCudaArray<T>::~ZCudaArray() 
{
	free();
}

template <typename T>
void
ZCudaArray<T>::free()
{
	_freeDevice();
	_freeHost();

	_init();
}

template <typename T>
void
ZCudaArray<T>::_freeHost()
{
	if( !_hostPtr ) { return; }

	if( _pinned ) { zCheckError( cudaFreeHost( _hostPtr ) ); }
	else          { delete[] _hostPtr; }
	_hostPtr = (T*)NULL;
}

template <typename T>
void
ZCudaArray<T>::_freeDevice()
{
	if( !_devicePtr ) { return; }

	zCheckError( cudaFree( _devicePtr ) );
	_devicePtr = (T*)NULL;
}

template <typename T>
bool
ZCudaArray<T>::setLength( int n, bool pinned )
{
	if( n <= 0 ) { free(); return true; }
	if( n == _size ) { return true; }
	_size = n;
	if( !_allocHost( pinned ) ) { return false; }
	if( !_allocDevice() ) { return false; }
	return true;
}

template <typename T>
bool
ZCudaArray<T>::_allocHost( bool pinned )
{
	_freeHost();
	if( !_size ) { return true; }

	if( _pinned ) {

		cudaError_t stat = cudaHostAlloc( (void**)&_hostPtr, _size*sizeof(T), 0 );
		if( !zCheckError(stat) ) { return false; }

	} else {

		try { _hostPtr = new T[_size]; }
		catch( std::bad_alloc& e ) { return false; }
	}

	return true;
}

template <typename T>
bool
ZCudaArray<T>::_allocDevice()
{
	_freeDevice();
	if( !_size ) { return true; }

	cudaError_t stat = cudaMalloc( (void**)&_devicePtr, _size*sizeof(T) );
	if( !zCheckError( stat ) ) { return false; }

	return true;
}

template <typename T>
bool
ZCudaArray<T>::copy( Direction dir )
{
	switch( dir )
	{
		case HOST_TO_DEVICE: { if( !_copyHostToDevice() ) { return false; } break; }
		case DEVICE_TO_HOST: { if( !_copyDeviceToHost() ) { return false; } break; }
		default: {}
	}

	return true;
}

template <typename T>
bool
ZCudaArray<T>::_copyHostToDevice()
{
	if( !_hostPtr ) { return true; }
	if( !_devicePtr ) { if( !_allocDevice() ) { return false; } }
	cudaError_t stat = cudaMemcpy( _devicePtr, _hostPtr, _size*sizeof(T), cudaMemcpyHostToDevice );
	return zCheckError( stat );
}

template <typename T>
bool
ZCudaArray<T>::_copyDeviceToHost()
{
	if( !_devicePtr ) { return true; }
	if( !_hostPtr ) { if( !_allocHost( _pinned ) ) { return false; } }
	cudaError_t stat = cudaMemcpy( _hostPtr, _devicePtr, _size*sizeof(T), cudaMemcpyDeviceToHost );
	return zCheckError( stat );
}

template <typename T>
inline T*
ZCudaArray<T>::hostPtr()
{
	return _hostPtr;
}

template <typename T>
inline T*
ZCudaArray<T>::devicePtr()
{
	return _devicePtr;
}

template <typename T>
inline int
ZCudaArray<T>::length() const 
{
	return _size;
}

#endif

