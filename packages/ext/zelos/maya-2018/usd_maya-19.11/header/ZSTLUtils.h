//-------------//
// ZSTLUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.24                               //
//-------------------------------------------------------//

#ifndef _ZSTLUtils_h_
#define _ZSTLUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

template <class T>
int ZRemove( std::vector<T>& v, const std::list<int>& l )
{
	const int vSize = (int)v.size();
	if( vSize == 0 ) { return 0; }

	const int lSize = (int)l.size();
	if( lSize == 0 ) { return 0; }

	std::vector<bool> mask( vSize, false );

	int numToDelete = 0;
	std::list<int>::const_iterator lItr = l.begin();
	for( ; lItr != l.end(); ++lItr )
	{
		const int idx = *lItr;
		if( idx >= vSize ) { continue; }
		mask[idx] = true;
		++numToDelete;
	}

	if( numToDelete == vSize )
	{
		v.clear();
		return 0;
	}

	const int finalSize = vSize - numToDelete;

	std::vector<T> tmpV( finalSize );

	for( int i=0, count=0; i<vSize; ++i )
	{
		if( mask[i] ) { continue; }
		tmpV[count++] = v[i];
	}

	v.swap( tmpV );

	return finalSize;
}

template <class T>
int ZRemoveRedundancy( std::vector<T>& v )
{
	std::set<T> s;

	typename std::vector<T>::const_iterator vItr = v.begin();
	for( ; vItr!=v.end(); ++vItr )
	{
		s.insert( *vItr );
	}

	const int size = (int)s.size();
	v.resize( size );

	std::copy( s.begin(), s.end(), v.begin() );

	return size;
}

template <class T>
void ZInsert( std::vector<T>& a, int index, T e )
{
	const int N = (int)a.size();
	a.push_back( a.back() );
	for( int i=N-1; i>index; --i )
	{
		a[i] = a[i-1];
	}
	a[index] = e;
}

template <class T>
void ZErase( std::vector<T>& a, int index )
{
	const int N = (int)a.size();
	for( int i=index; i<N-1; ++i )
	{
		a[i] = a[i+1];
	}
	a.pop_back();
}

template <class T>
void ZZeroize( std::vector<T>& a )
{
	if( a.size() > 0 )
	{
		memset( (char*)&a[0], 0, a.size()*sizeof(T) );
	}
}

template <typename T>
void ZPrint( std::vector<T>& v )
{
	int i = 0;
	typename std::vector<T>::const_iterator itr = v.begin();
	for( ; itr!=v.end(); ++itr, ++i )
	{
		cout << i << ": " << (*itr) << endl;
	}
}

template <typename T>
long double
ZDot( const std::vector<T>& x, const std::vector<T>& y )
{
	const int N = (int)x.size();
	long double sum = (T)0;
	for( int i=0; i<N; ++i )
	{
		sum += x[i] * y[i];
	}
	return sum;
}

template <typename T>
T
ZInfNorm( const std::vector<T>& x )
{
	const int N = (int)x.size();
	double maxVal = 0;
	for( int i=0; i<N; ++i )
	{
		const T absX = (x[i]>0)?(x[i]):(-x[i]);
		if( absX > maxVal )
		{
			maxVal = absX;
		}
	}
	return maxVal;
}

// y = alpha*x + y
template <typename T>
void
ZAddScaled( T alpha, const std::vector<T>& x, std::vector<T>& y )
{ 
	const int N = (int)x.size();
	for( int i=0; i<N; ++i ) { y[i] += alpha*x[i]; }
}

static void
ZEraseNewLineCharacter( std::string& s )
{
	s.erase( s.find_last_not_of(" \n\r") );
	//cf) s.erase(s.find_last_not_of(" \n\t\r"));
}

ZELOS_NAMESPACE_END

#endif

