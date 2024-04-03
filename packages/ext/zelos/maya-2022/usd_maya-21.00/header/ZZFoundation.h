//----------------//
// ZZFoundation.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.10                               //
//-------------------------------------------------------//

#ifndef _ZZFoundation_h_
#define _ZZFoundation_h_

#include <ZelosCudaBase.h>

inline bool zCheckError( cudaError_t stat )
{
	if( stat == cudaSuccess ) { return true; }
	cout << cudaGetErrorString(stat) << endl;
	cout << "file: " << __FILE__ << endl;
	cout << "line: " << __LINE__ << endl;
	return false;
}

// This will output the proper error string when calling cudaGetLastError
#define zGetLastCudaError(msg) _zGetLastCudaError( msg, __FILE__, __LINE__ )
inline void _zGetLastCudaError( const char* errorMessage, const char* file, const int line )
{
	cudaError_t err = cudaGetLastError();

	if( cudaSuccess != err )
	{
		cout << "Cuda Error @ " << file << "(" << line << ") : " << errorMessage << endl;
		cudaDeviceReset();
		exit( EXIT_FAILURE );
	}
}

struct zXYZ  { float x,y,z; };
struct zRGB  { float r,g,b; };
struct zRGBA { float r,g,b,a; };

#endif

