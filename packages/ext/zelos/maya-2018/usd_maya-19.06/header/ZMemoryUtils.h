//----------------//
// ZMemoryUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.24                               //
//-------------------------------------------------------//

#ifndef _ZMemoryUtils_h_
#define _ZMemoryUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

template <typename T>
inline bool
ZAlloc( T*& data, int length )
{
	if( length <= 0 ) { if( data ) { delete[] data; } }
	else if( data ) { delete[] data; }

	try {

		data = new T[length];

	} catch( std::bad_alloc& e ) {

		cout << "Error@ZAlloc(): Allocation failed (" << e.what() << ")." << endl;
		return false;

	}

	return true;
}

template <typename T>
inline void
ZFree( T*& data, bool isArray=true )
{
	if( isArray ) { if( data ) { delete[] data; } }
	else          { if( data ) { delete data;   } }
	data = (T*)NULL;
}

template <typename T>
inline void
ZSwap( T& a, T& b )
{
	T c=a; a=b; b=c;
}

template <typename T>
inline void
ZSwitchEndian( T& value )
{ 
	T temp = value;
	char* src  = (char*)&temp;
	char* dst = (char*)&value;
	const int size = sizeof(T);
	FOR( i, 0, size )
	{
		dst[i] = src[size-i-1];
	}
}

template <typename T>
inline bool
ZGetBit( T x, int N )
{
	return ( ( x & (1<<N) ) >> N );
}

template <typename T>
inline void
ZPrintBits( T x, int N )
{
	switch( N )
	{
		case  4: { cout << std::bitset< 4>(x) << endl; break; }
		case  8: { cout << std::bitset< 8>(x) << endl; break; }
		case 16: { cout << std::bitset<16>(x) << endl; break; }
		case 32: { cout << std::bitset<32>(x) << endl; break; }
		default: { break; }
	}
}

inline uint32_t
ZHowManyOnesInBits( uint32_t x )
{
	// count number of ones
	x = (x & 0x55555555) + ((x >> 1) & 0x55555555); // add pairs of bits
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333); // add bit pairs
	x = (x & 0x0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f); // add nybbles
	x += (x >> 8);                                  // add bytes
	x += (x >> 16);                                 // add words

	return(x & 0xff);
}

template <typename T>
inline void
ZSetBit( T& x, int N )
{
	x |= (1<<N);
}

template <typename T>
inline T
ZClearBit( T& x, int N )
{
	x &= ~(1<<N);
}

template <class T>
inline void
ZZeroize( T& v )
{
	char* x = (char*)&v;
	const int size = (int)sizeof(T);
	FOR(i,0,size) { x[i]=0; }
}

ZELOS_NAMESPACE_END

#endif

