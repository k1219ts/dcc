//----------------------//
// ZSparseMatrixUtils.h //
//-------------------------------------------------------//
// author: Inyong Jeon @ Seoul National Univ.            //
//         Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
//         Nayoung Kim @ Dexter Studios                  //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZSparseMatrixUtils_h_
#define _ZSparseMatrixUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// Ax = b
template <int N, class T>
bool
Multiply( ZSparseMatrix<T>& A, ZTuple<N,T>& x, ZTuple<N,T>& b, bool useOpenMP=true )
{
	const int M = A.m(); // = N (A must be a square matrix.)

	if( M != x.length() ) { return false; }
	if( M != b.length() ) { return false; }

	b.zeroize();

	#pragma omp parallel for if( useOpenMP )
	for( int i=0; i<M; ++i )
	{
		const int& r0 = A.r[i];
		const int& r1 = A.r[i+1];

		for( int j=r0; j<r1; ++j )
		{
			b[i] += A.v[j] * x[ A.c[j] ];
		}
	}

	return true;
}

inline bool
Multiply( const ZSparseMatrix<float>& A, const ZFloatArray& x, ZFloatArray& b, bool useOpenMP=true )
{
    const int M = A.m();
    if( M != x.length() ) return false;
    
    if( b.length() != M ) b.setLength(M);
    b.zeroize();
    
	#pragma omp parallel for if( useOpenMP )
	for( int i=0; i<M; ++i )
	{
		const int& r0 = A.r[i];
		const int& r1 = A.r[i+1];
        
        float buff = 0.0;
		for( int j=r0; j<r1; ++j )
		{
            if(A.c[j] != -1) buff += A.v[j] * x[A.c[j]];
		}
        
        b[i] = buff;
	}

	return true;    
}

inline void
InverseDiagonalMultiply( const ZSparseMatrix<float>& A, const ZFloatArray& x, ZFloatArray& b, const bool useOpenMP=true )
{
    const int M = A.m();
    if( M != x.length() ) return;
    
    if( b.length() != M ) b.setLength(M);
    b.zeroize();
    
	#pragma omp parallel for if( useOpenMP )
	for( int i=0; i<M; ++i )
	{
		const int& r0 = A.r[i];
		const int& r1 = A.r[i+1];
        
		for( int j=r0; j<r1; ++j )
		{
            if(i == A.c[j] && 0.0 != A.v[j]) 
            {
                b[i] = (1.0/A.v[j]) * x[i];
                break;                
            }
		}
	}    
}

// warning: comparison between signed and unsigned integer expressions
//inline bool
//Multiply( const ZFloatArray& a, const ZFloatArray& b, float& result, const bool useOpenMP=true )
//{
//    const int numThreads = useOpenMP ? std::min(omp_get_num_procs(), a.length()) : 1;
//    
//    std::vector<float> buffer; 
//    buffer.resize(numThreads);
//    std::fill(buffer.begin(), buffer.end(), 0.0);
//    
//	#pragma omp parallel for if( useOpenMP )
//	for( int i=0; i<a.length(); i++ )
//	{
//        const int tid = omp_get_thread_num();  
//        buffer[tid] += a[i]*b[i];               
//	}
// 
//    result = 0.0;        
//    for( int i=0; i<buffer.size(); i++ )
//    {
//        result += buffer[i];
//    }    
//    
//    return true;    
//}

inline void
Multiply( const ZFloatArray& a, const ZFloatArray& b, float& result, const bool useOpenMP=true )
{
	const int N = a.length();

	result = 0.f;

	const int numThreads = useOpenMP ? ZMin( omp_get_num_procs(), a.length() ) : 1;

	ZFloatArray buffer( numThreads );

	#pragma omp parallel for if( useOpenMP )
	for( int i=0; i<N; ++i )
	{
		const int tid = omp_get_thread_num();
		buffer[tid] += a[i]*b[i];
	}

	for( int i=0; i<numThreads; ++i )
	{
		result += buffer[i];
	}    
}

inline void 
Multiply( const ZFloatArray& a, const float& b, ZFloatArray& c, const bool useOpenMP=true )
{
	const int N = a.length();

	c.setLength( N, false );

	#pragma omp parallel for if( useOpenMP )
	for( int i=0; i<N; ++i )
	{
		c[i] = a[i] * b;
	}
}

inline void
Equal( const ZFloatArray& a, ZFloatArray& c, bool useOpenMP=true )
{
	const int N = a.length();

	c.setLength( N, false );

	#pragma omp parallel for if( useOpenMP )
	for(int i=0; i<N; ++i )
	{
		c[i] = a[i];
	}
}

inline void 
Subtract( const ZFloatArray& a, const ZFloatArray& b, ZFloatArray& c, const bool useOpenMP=true )
{
	const int N = a.length();

	c.setLength( N, false );

	#pragma omp parallel for if( useOpenMP )
	for( int i=0; i<N; ++i )
	{
		c[i] = a[i] - b[i];
	}
}

inline void
Add( const ZFloatArray& a, const ZFloatArray& b, ZFloatArray& c, const bool useOpenMP=true )
{
	const int N = a.length();

	c.setLength( N, false );

	#pragma omp parallel for if( useOpenMP )
	for( int i=0; i<N; ++i )
	{
		c[i] = a[i] + b[i];
	}
}

// Ax = b
template <class T>
bool
Multiply( const ZSparseMatrix<ZDenseMatrix<3,3,T> >& A, const ZVectorArray& x, ZVectorArray& b, const bool useOpenMP=true )
{
	const int M = A.m(); // = N (A must be a square matrix.)

	if( M != x.length() ) { return false; }
	if( M != b.length() ) { return false; }

	b.zeroize();

	#pragma omp parallel for if( useOpenMP )
	for( int i=0; i<M; ++i )
	{
		const int& r0 = A.r[i];
		const int& r1 = A.r[i+1];

		for( int j=r0; j<r1; ++j )
		{
			b[i] += A.v[j] * x[ A.c[j] ];
		}
	}

	return true;
}

template <class T>
bool
Multiply( std::vector< ZTuple<3,T> >& b, const ZSparseMatrix< ZDenseMatrix<3,3,T> >& A, const std::vector< ZTuple<3,T> >& x, bool useOpenMP=true )
{
	const int N = A.n();
	const int M = A.n();

	if( ( N != M ) || ( N != (int)x.size() ) || ( N != (int)b.size() ) )
	{
		cout << "Error@Multiply(): Invalid dimension." << endl;
		return false;
	}

	memset( &(b[0]), 0, sizeof(T)*N*3 );

	#pragma omp parallel for if( useOpenMP )
	for( int i=0; i<M; ++i )
	{
		const int& r0 = A.r[i];
		const int& r1 = A.r[i+1];

		for( int j=r0; j<r1; ++j )
		{
			b[i] += A.v[j] * x[ A.c[j] ];
		}
	}

	return true;
}


// A=B*C. Only supports same data type for now...
template <class T>
bool
Multiply( ZSparseMatrix<T>& A, const ZSparseMatrix<T>& B, const ZSparseMatrix<T>& C, bool useOpenMP=true )
{
	// 1. Dimension check.
	// A(m x p) = B(m x n) * C(n x p)
	const int NB = B.n();
	const int MC = C.m();
	if( NB != MC )
	{
		cout << "Error@Multiply(): Invalid dimension." << endl;
		return false;
	}

	// 2. Set the memory for the matrix A(m x p).
	// 3. Calculation (Results are stored in triplet array.)
	// A(i,j) = SUM( Bi0*C0j + Bi1*C1j + ... + Bin*Cnj )
	const int MB = B.m();
	const int NC = C.n();

	T val = 0;
	T valB = 0;
	T valC = 0;
//	int nnzA = 0;		// just for checking..

	std::vector<std::list<int> > AIJs; AIJs.resize( MB );
	ZVectorArray triplet;

	FOR( i, 0, MB )
	{
		// i-th row of matrix B.
		FOR( k, 0, NC )
		{
			// k-th col of matrix C.
			T sum = 0;

			bool BIJexist = false;
			bool CIJexist = false;

			#pragma omp parallel for if( useOpenMP )
			FOR( j, 0, MC )
			{
				// j-th col of matrix B.

				BIJexist = B.getValue( i, j, valB );
				if( !BIJexist ) { valB = 0; }

				CIJexist = C.getValue( j, k, valC );
				if( !CIJexist ) { valC = 0; }

				val = valB * valC;
				sum += val;
			}

			AIJs[i].push_back( k );
			triplet.push_back( ZVector( i, k, sum ) );
//			++nnzA;		// just for checking..
		}
	}

	A.set( AIJs );

	const int MA = A.m();
	std::vector<int>& r = A.r;

	// 4. Fill the entries. (From triplet to v array)
	FOR( i, 0, MA )
	{
		const int& r0 = r[i];
		const int& r1 = r[i+1];

		FOR( j, r0, r1 )
		{
			ZVector& t = triplet[j];
			A.setValue( t[0], t[1], t[2] );
		}
	}

	return true;
}

ZELOS_NAMESPACE_END

#endif

