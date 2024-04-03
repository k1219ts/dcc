//------------//
// ZCudaVbo.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2013.06.20                               //
//-------------------------------------------------------//

#ifndef _ZCudaVbo_h_
#define _ZCudaVbo_h_

#include <ZelosCudaBase.h>

// Copyright 1993-2012 NVIDIA Corporation. All rights reserved.
// GpuArray class is renamed to ZCudaVbo. 
// (This VBO has the interoperability to CUDA).
template <class T> 
class ZCudaVbo
{
	private:

		GLuint                _id;
		int                   _size;
		T*                    _hostPtr;
		T*                    _devicePtr;
		cudaGraphicsResource* _resource;		// resources OpenGL-CUDA exchange

	public:

		enum Direction { HOST_TO_DEVICE, DEVICE_TO_HOST };

	public:

		ZCudaVbo();
		ZCudaVbo( int n );

		~ZCudaVbo();

		void free();

		void setLength( int n, bool elementArray=false );

		bool map();    // Map vbo before getting device ptr
		bool unmap();  // Unmap vbo after using device ptr

		bool copy( Direction dir );

		GLuint id() const;

		T* hostPtr();
		T* devicePtr();

		int length() const;

	private:

		void _init();

		bool _allocHost();
		bool _allocDevice( bool elementArray );

		void _freeHost();
		void _freeDevice();
};

template <class T> 
ZCudaVbo<T>::ZCudaVbo()
{
	_init();
}

template <class T> 
ZCudaVbo<T>::ZCudaVbo( int n )
{
	_init();
	setLength( n );
}

template <class T>
void
ZCudaVbo<T>::_init()
{
	_id        = 0;
	_size      = 0;
	_hostPtr   = (T*)NULL; 
	_devicePtr = (T*)NULL;
	_resource  = (cudaGraphicsResource*)NULL;
}

template <class T> 
ZCudaVbo<T>::~ZCudaVbo()
{ 
	free();
}

template <class T> 
void
ZCudaVbo<T>::free()
{
	_freeHost();
	_freeDevice();

	_init();
}

template <class T> 
void
ZCudaVbo<T>::_freeHost()
{
	if( !_hostPtr ) { return; }

	delete[] _hostPtr;
	_hostPtr = (T*)NULL;
}

template <class T> 
void
ZCudaVbo<T>::_freeDevice()
{
	if( !_id ) { return; }

	glBindBuffer( 1, _id );
	glDeleteBuffers( 1, &_id );
	zCheckError( cudaGraphicsUnregisterResource( _resource ) );

	_id = 0;
	_devicePtr = (T*)NULL;
	_resource = (cudaGraphicsResource*)NULL;
}

template <class T> 
void
ZCudaVbo<T>::setLength( int n, bool elementArray )
{
	if( n <= 0 ) { free(); return; }
	if( n == _size ) { return; }
	_size = n;
	if( !_allocHost() ) { free(); return; }
	if( !_allocDevice( elementArray ) ) { free(); return; }
}

template <class T> 
bool
ZCudaVbo<T>::_allocHost()
{
	_freeHost();
	if( !_size ) { return true; }

	try { _hostPtr = new T[_size]; }
	catch( std::bad_alloc& e ) { return false; }

	return true;
}

template <class T> 
bool
ZCudaVbo<T>::_allocDevice( bool elementArray )
{
	_freeDevice();
	if( !_size ) { return true; }

	glGenBuffers( 1, &_id );

	const int size = _size * sizeof(T);

	if( elementArray ) {

		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _id );
		glBufferData( GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );

	} else {

		glBindBuffer( GL_ARRAY_BUFFER, _id );
		glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );

	}

	// [cudaGraphicsGLRegisterBuffer]
	// Registers the buffer object for access by CUDA.
	// _resource: A returned resource to the registered object.
	// flag: cudaGraphicsRegisterFlagsNone         : Specifies no hints about how this resource will be used.
	//                                               Therefore, it is assumed that this resource will be read from and written to by CUDA.
	//       cudaGraphicsRegisterFlagsReadOnly     : Specifies that CUDA will not write to this resource.
	//       cudaGraphicsRegisterFlagsWriteDiscard : Specifies that CUDA will not read from this resource and will write over the entire contents of the resource.
	//                                               So, none of the data previously stored in the resource will be preserved.
	cudaError_t stat = cudaGraphicsGLRegisterBuffer( &_resource, _id, cudaGraphicsRegisterFlagsNone );
	return zCheckError( stat );
}

template <class T> 
bool
ZCudaVbo<T>::map()
{
	if( !_id ) { return true; }

	// [cudaGraphicsMapResources]
	// Maps the count graphics resources in resources for access by CUDA.
	// The resources in resources may be accessed by CUDA until they are unmapped.
	cudaError_t stat = cudaGraphicsMapResources( 1, &_resource, 0 );
	if( !zCheckError( stat ) ) { return false; }

	// [cudaGraphicsResourceGetMappedPointer]
	// Returns in _devicePtr a pointer through which the mapped graphics resource resource may be accessed.
	// _bytes: The returned size of the momory in bytes.
	size_t bytes = 0;
	stat = cudaGraphicsResourceGetMappedPointer( (void**)&_devicePtr, &bytes, _resource );
	if( !zCheckError( stat ) ) { return false; }

	return true;
}

template <class T> 
bool
ZCudaVbo<T>::unmap()
{
	if( !_id ) { return true; }

	// [cudaGraphicsUnmapResources]
	// Unmaps the count graphics resources in resources.
	// Once unmapped, the resources in resources may not be accessed by CUDA until they ara mapped again.
	cudaError_t stat = cudaGraphicsUnmapResources( 1, &_resource, 0 );
	if( !zCheckError( stat ) ) { return false; }
	_devicePtr = (T*)NULL;

	return true;
}

template <class T> 
bool
ZCudaVbo<T>::copy( Direction dir )
{
	if( !map() ) { return false; }

	switch( dir )
	{
		case HOST_TO_DEVICE:
		{
			cudaError_t stat = cudaMemcpy( (void*)_devicePtr, (void*)_hostPtr, _size*sizeof(T), cudaMemcpyHostToDevice );
			if( !zCheckError( stat ) ) { return false; }
			break;
		}

		case DEVICE_TO_HOST:
		{
			cudaError_t stat = cudaMemcpy( (void*)_hostPtr, (void*)_devicePtr, _size*sizeof(T), cudaMemcpyDeviceToHost );
			if( !zCheckError( stat ) ) { return false; }
			break;
		}

		default: {}
	}

	if( !unmap() ) { return false; }

	return true;
}

template <class T> 
inline GLuint
ZCudaVbo<T>::id() const
{
	return _id;
}

template <class T> 
inline T*
ZCudaVbo<T>::hostPtr()
{
	return _hostPtr;
}

template <class T> 
inline T*
ZCudaVbo<T>::devicePtr()
{
	return _devicePtr;
}

template <class T> 
inline int
ZCudaVbo<T>::length() const
{
	return _size;
}

#endif

