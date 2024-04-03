//-----------------//
// ZSparseMatrix.h //
//-------------------------------------------------------//
// author: Inyong Jeon @ Seoul National Univ.            //
//         Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
//         Nayoung Kim @ Dexter Studios                  //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZSparseMatrix_h_
#define _ZSparseMatrix_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// ex)
//
// [1 0 0 0] 
// [2 3 0 0] 
// [0 4 5 0]
// [6 0 7 8] 
//
// T   v[] = { 1, 2,3, 4,5, 6,7,8 }: size = non-zero entries
// int r[] = { 0, 1, 3, 5, 8 }     : size = m+1
// int c[] = { 0, 0,1, 1,2, 0,2,3 }: size = non-zero entries

/// @brief MxN sparse matrix.
/**
	Compressed Sparse Row (CSR) based mxn sparse matrix (in most cases, m=n: square matrix)
*/
template <class T>
class ZSparseMatrix
{
	private:

		int _m;				// the number of rows
		int _n;				// the number of columns
		int _nnz;			// the number of non-zero entres

	public: // but, they must be treated carefully like like as private members.

		// The CSR format stores a matrix using three one-dimensional arrays.
		std::vector<T>   v;	// v[i]: the value of the i-th nonzero entry
		std::vector<int> r;	// r[i]: the array index of the first nonzero entry of the i-th row
		std::vector<int> c;	// c[i]: the column index of the i-th nonzero entry

	public: 

		ZSparseMatrix();
		ZSparseMatrix( int numRows, int numColumns, int nonzeroElementsCount );
		ZSparseMatrix( const ZSparseMatrix& A );

		void reset();

		void zeroize();

		void set( int numRows, int numColumns, int nonzeroElementsCount );
		void set( const std::vector<std::list<int> >& IJs );	// (i,j) reserve the spots. list holds col #.
		void set( const ZSparseMatrix& A, bool withOutValues=true );

		void setFivePointLaplacian ( const int Nx, const int Ny );
		void setSevenPointLaplacian( const int Nx, const int Ny, const int Nz );

		const T& operator()( const int& i, const int& j ) const;
		T& operator()( const int& i, const int& j );

		ZSparseMatrix& operator=( const ZSparseMatrix& A );

		ZSparseMatrix& operator+=( const ZSparseMatrix& A );
		ZSparseMatrix& operator-=( const ZSparseMatrix& A );

		ZSparseMatrix& operator*=( const int& s );
		ZSparseMatrix& operator*=( const float& s );
		ZSparseMatrix& operator*=( const double& s );

		ZSparseMatrix& operator/=( const float& s );
		ZSparseMatrix& operator/=( const double& s );

		ZSparseMatrix& transpose();
		ZSparseMatrix  transposed() const;

		bool getValue( const int& i, const int& j, T& value ) const;
		bool setValue( const int& i, const int& j, const T& value );

		bool setValueInc( const int& i, const int& j, const T& value );
		bool setValueMul( const int& i, const int& j, const T& value );

		int getArrayIndex( const int& i, const int& j ) const;

        void addRow( const std::vector<T> val, const std::vector<int> col );

		int m() const;   // the number of rows
		int n() const;   // the number of columns
		int nnz() const; // the number of non-zero entres

		bool isValid( const int& i, const int& j ) const;
		bool isNonZero( const int& i, const int& j ) const;

		bool isSymmetric( float tolerance=Z_EPS ) const;
		bool isBlockSymmetric( float tolerance=Z_EPS ) const;

		void print() const;	// print the matrix at the terminal.

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

template <class T>
ZSparseMatrix<T>::ZSparseMatrix()
: _m(0), _n(0), _nnz(0)
{
}

template <class T>
ZSparseMatrix<T>::ZSparseMatrix( int numRows, int numColumns, int nonzeroElementsCount )
{
	set( numRows, numColumns, nonzeroElementsCount );
}

template <class T>
ZSparseMatrix<T>::ZSparseMatrix( const ZSparseMatrix<T>& A )
{
	*this = A;
}

template <class T>
void
ZSparseMatrix<T>::reset()
{
	_m   = 0;
	_n   = 0;
	_nnz = 0;

	v.clear();
	r.clear();
	c.clear();
}

template <class T>
void
ZSparseMatrix<T>::zeroize()
{
	memset( &v[0], 0, _nnz*sizeof(T) );
}

template <class T>
void
ZSparseMatrix<T>::set( int numRows, int numColumns, int nonzeroElementsCount )
{
	if( ( _m != numRows ) || ( _n != numColumns ) || ( _nnz!=nonzeroElementsCount ) )
	{
		_m   = numRows;
		_n   = numColumns;
		_nnz = nonzeroElementsCount;

		v.resize( _nnz );
		r.resize( _m+1 );
		c.resize( _nnz );
	}

	memset( &v[0], 0, sizeof(T)  *(_nnz) );
	memset( &r[0], 0, sizeof(int)*(_m+1) );
	memset( &c[0], 0, sizeof(int)*(_nnz) );
}

template <class T>
void
ZSparseMatrix<T>::set( const std::vector<std::list<int> >& IJs )
{
	const int numRows = (int)IJs.size();

	int nonZeroCount = 0;
	{
		for( int i=0; i<numRows; ++i )
		{
			std::list<int>::const_iterator itr = IJs[i].begin();

			for( ; itr != IJs[i].end(); ++itr )
			{
				++nonZeroCount;
			}
		}
	}

	set( numRows, numRows, nonZeroCount );

	int cntNNZ = 0;

	for( int i=0; i<numRows; ++i )
	{
		r[i] = cntNNZ;

		std::list<int>::const_iterator itr = IJs[i].begin();
		for( ; itr != IJs[i].end(); ++itr )
		{
			c[cntNNZ] = (*itr);	
			++cntNNZ;
		}
	}

	r[numRows] = cntNNZ;
}

template <class T>
void
ZSparseMatrix<T>::set( const ZSparseMatrix<T>& A, bool withoutValues )
{
	if( ( _m != A.m() ) || ( _n != A.n() ) || ( _nnz!=A.nnz() ) )
	{
		_m   = A.m();
		_n   = A.n();
		_nnz = A.nnz();

		v.resize( A.nnz() );
		r.resize( A.m()+1 );
		c.resize( A.nnz() );
	}

	if( !withoutValues )
	{
		v.assign( A.c.begin(), A.c.end() );
	}

	r.assign( A.r.begin(), A.r.end() );
	c.assign( A.c.begin(), A.c.end() );
}


template <class T>
void
ZSparseMatrix<T>::setFivePointLaplacian( const int Nx, const int Ny )
{
	_nnz = Nx * Ny * 5;
	_m = _n = Nx * Ny;
	
	v.resize( _nnz );
	c.resize( _nnz );
	r.resize( _m+1 );

	for(int i=0; i<_m; i++)
	{
		const int ix = i*5;
		r[i] = ix;				
		
		c[ ix+0 ] = i;
		c[ ix+1 ] = i+1;
		c[ ix+2 ] = i-1;
		c[ ix+3 ] = i+Nx;
		c[ ix+4 ] = i-Nx;
		
		v[ ix+0 ] = (T)0;
		v[ ix+1 ] = (T)0;
		v[ ix+2 ] = (T)0;
		v[ ix+3 ] = (T)0;
		v[ ix+4 ] = (T)0;
	}
	
	r[_m] = _m * 5;
}

template <class T>
void
ZSparseMatrix<T>::setSevenPointLaplacian( const int Nx, const int Ny, const int Nz )
{
	_nnz = Nx * Ny * Nz * 7;
	_m = _n = Nx * Ny * Nz;
	
	int Nxy = Nx * Ny;

	v.resize( _nnz );
	c.resize( _nnz );	
	r.resize( _m+1 );
	
	for(int i=0; i<_m; i++)
	{
		const int ix = i*7;
		r[i] = ix;
		
		c[ ix+0 ] = i;
		c[ ix+1 ] = i+1;
		c[ ix+2 ] = i-1;
		c[ ix+3 ] = i+Nx;
		c[ ix+4 ] = i-Nx;
		c[ ix+5 ] = i+Nxy;
		c[ ix+6 ] = i-Nxy;													
		
		v[ ix+0 ] = (T)0;
		v[ ix+1 ] = (T)0;
		v[ ix+2 ] = (T)0;
		v[ ix+3 ] = (T)0;
		v[ ix+4 ] = (T)0;
		v[ ix+5 ] = (T)0;
		v[ ix+6 ] = (T)0;		
	}
	
	r[_m] = _m * 7;	
}

template <class T>
const T&
ZSparseMatrix<T>::operator()( const int& i, const int& j ) const
{
	const int& r0 = r[i];
	const int& r1 = r[i+1];

	if( i > j ) { // searching in the lower part

		for( int k=r0; k<r1; ++k )
		{
			if( c[k] == j )
			{
				return v[k];
			}
		}

	} else { // searching in the upper part

		for( int k=r1-1; k>=r0; --k )
		{
			if( c[k] == j )
			{
				return v[k];
			}
		}

	}

	return v[0]; // but, meaningless return
}

template <class T>
T&
ZSparseMatrix<T>::operator()( const int& i, const int& j )
{
	const int& r0 = r[i];
	const int& r1 = r[i+1];

	if( i > j ) { // searching in the lower part

		for( int k=r0; k<r1; ++k )
		{
			if( c[k] == j )
			{
				return v[k];
			}
		}

	} else { // searching in the upper part

		for( int k=r1-1; k>=r0; --k )
		{
			if( c[k] == j )
			{
				return v[k];
			}
		}

	}

	return v[0]; // but, meaningless return
}

template <class T>
ZSparseMatrix<T>&
ZSparseMatrix<T>::operator=( const ZSparseMatrix<T>& A )
{
	_m   = A._m;
	_n   = A._n;
	_nnz = A._nnz;

	v.resize( _nnz );
	r.resize( _m+1 );
	c.resize( _nnz );

	v.assign( A.v.begin(), A.v.end() );
	r.assign( A.r.begin(), A.r.end() );
	c.assign( A.c.begin(), A.c.end() );

	return (*this);
}

template <class T>
ZSparseMatrix<T>&
ZSparseMatrix<T>::operator+=( const ZSparseMatrix& A )
{
	assert( _m   == A._m   );
	assert( _n   == A._n   );
	assert( _nnz == A._nnz );

	assert( v.size() == A.v.size() );
	assert( r.size() == A.r.size() );
	assert( c.size() == A.c.size() );

	FOR( i, 0, _nnz )
	{
		v[i] += A.v[i];
	}

	return (*this);
}

template <class T>
ZSparseMatrix<T>&
ZSparseMatrix<T>::operator-=( const ZSparseMatrix& A )
{
	assert( _m   == A._m   );
	assert( _n   == A._n   );
	assert( _nnz == A._nnz );

	assert( v.size() == A.v.size() );
	assert( r.size() == A.r.size() );
	assert( c.size() == A.c.size() );

	FOR( i, 0, _nnz )
	{
		v[i] -= A.v[i];
	}

	return (*this);
}

template <class T>
ZSparseMatrix<T>&
ZSparseMatrix<T>::operator*=( const int& s )
{
	FOR( i, 0, _nnz )
	{
		v[i] *= s;
	}

	return (*this);
}

template <class T>
ZSparseMatrix<T>&
ZSparseMatrix<T>::operator*=( const float& s )
{
	FOR( i, 0, _nnz )
	{
		v[i] *= s;
	}

	return (*this);
}

template <class T>
ZSparseMatrix<T>&
ZSparseMatrix<T>::operator*=( const double& s )
{
	FOR( i, 0, _nnz )
	{
		v[i] *= s;
	}

	return (*this);
}

template <class T>
ZSparseMatrix<T>&
ZSparseMatrix<T>::operator/=( const float& s )
{
	const float _s = 1.f / ( s + Z_EPS );

	FOR( i, 0, _nnz )
	{
		v[i] *= _s;
	}

	return (*this);
}

template <class T>
ZSparseMatrix<T>&
ZSparseMatrix<T>::operator/=( const double& s )
{
	const double _s = 1.0 / ( s + Z_EPS );

	FOR( i, 0, _nnz )
	{
		v[i] *= _s;
	}

	return (*this);
}

template <class T>
ZSparseMatrix<T>&
ZSparseMatrix<T>::transpose()
{
	const int m   = (*this).m();
	const int n   = (*this).n();
	const int nnz = (*this).nnz();

	std::vector<T>&   v = (*this).v;
	std::vector<int>& r = (*this).r;
	std::vector<int>& c = (*this).c;

	std::vector<std::list<int> > JIs; JIs.resize( n );	// transposed matrix.
	ZVectorArray triplet( nnz );						// store the (i,j,val)

	// memory allocation.
	FOR( i, 0, m )
	{
		// i-th row.
		const int& r0 = r[i];
		const int& r1 = r[i+1];

		// fill the transposed JIs. (i,j)=>(j,i)
		FOR( j, r0, r1 )
		{
			ZVector& t = triplet[j];

			// set the JIs.
			JIs[c[j]].push_back( i );

			// store the (i,j,val).
			t[0] = c[j];
			t[1] = i;
			t[2] = v[j];
		}
	}

	// reset the matrix.
	(*this).reset();
	(*this).set( n, m, nnz );
	(*this).set( JIs );

	// set values.
	FOR( i, 0, n )
	{
		// i-th row.
		const int& r0 = r[i];
		const int& r1 = r[i+1];

		// set the values in the right position.
		FOR( j, r0, r1 )
		{
			ZVector& t = triplet[j];
			(*this).setValue( t[0], t[1], t[2] );
		}
	}

	return (*this);
}

template <class T>
ZSparseMatrix<T>
ZSparseMatrix<T>::transposed() const
{
	ZSparseMatrix<T> tmp(*this);
	tmp.transpose();
	return tmp;
}

template <class T>
bool
ZSparseMatrix<T>::getValue( const int& i, const int& j, T& value ) const
{
	const int& r0 = r[i];
	const int& r1 = r[i+1];

	if( i > j ) { // searching in the lower part

		for( int k=r0; k<r1; ++k )
		{
			if( c[k] == j )
			{
				value = v[k];
				return true; // (i,j)-entry exists!
			}
		}

	} else { // searching in the upper part

		for( int k=r1-1; k>=r0; --k )
		{
			if( c[k] == j )
			{
				value = v[k];
				return true; // (i,j)-entry exists!
			}
		}

	}

	return false; // (i,j)-entry doesn't exist!
}

template <class T>
bool
ZSparseMatrix<T>::setValue( const int& i, const int& j, const T& value )
{
	const int& r0 = r[i];
	const int& r1 = r[i+1];

	if( i > j ) { // searching in the lower part

		for( int k=r0; k<r1; ++k )
		{
			if( c[k] == j )
			{
				v[k] = value;
				return true; // (i,j)-entry exists!
			}
		}

	} else { // searching in the upper part

		for( int k=r1-1; k>=r0; --k )
		{
			if( c[k] == j )
			{
				v[k] = value;
				return true; // (i,j)-entry exists!
			}
		}

	}

	return false; // (i,j)-entry doesn't exist!
}

template <class T>
bool
ZSparseMatrix<T>::setValueInc( const int& i, const int& j, const T& value )
{
	const int& r0 = r[i];
	const int& r1 = r[i+1];

	if( i > j ) { // searching in the lower part

		for( int k=r0; k<r1; ++k )
		{
			if( c[k] == j )
			{
				v[k] += value;
				return true; // (i,j)-entry exists!
			}
		}

	} else { // searching in the upper part

		for( int k=r1-1; k>=r0; --k )
		{
			if( c[k] == j )
			{
				v[k] += value;
				return true; // (i,j)-entry exists!
			}
		}

	}

	return false; // (i,j)-entry doesn't exist!
}

template <class T>
bool
ZSparseMatrix<T>::setValueMul( const int& i, const int& j, const T& value )
{
	const int& r0 = r[i];
	const int& r1 = r[i+1];

	if( i > j ) { // searching in the lower part

		for( int k=r0; k<r1; ++k )
		{
			if( c[k] == j )
			{
				v[k] *= value;
				return true; // (i,j)-entry exists!
			}
		}

	} else { // searching in the upper part

		for( int k=r1-1; k>=r0; --k )
		{
			if( c[k] == j )
			{
				v[k] *= value;
				return true; // (i,j)-entry exists!
			}
		}

	}

	return false; // (i,j)-entry doesn't exist!
}

template <class T>
int
ZSparseMatrix<T>::getArrayIndex( const int& i, const int& j ) const
{
	const int& r0 = r[i];
	const int& r1 = r[i+1];

	if( i > j ) { // searching in the lower part

		for( int k=r0; k<r1; ++k )
		{
			if( c[k] == j )
			{
				return k;
			}
		}

	} else { // searching in the upper part

		for( int k=r1-1; k>=r0; --k )
		{
			if( c[k] == j )
			{
				return k;
			}
		}

	}

	return -1; // but, meaningless return
}

template <class T>
inline int
ZSparseMatrix<T>::m() const
{
	return _m;
}

template <class T>
inline int
ZSparseMatrix<T>::n() const
{
	return _n;
}

template <class T>
inline int
ZSparseMatrix<T>::nnz() const
{
	return _nnz;
}

template <class T>
bool
ZSparseMatrix<T>::isValid( const int& i, const int& j ) const
{
	if( i <   0 ) { return false; }
	if( j <   0 ) { return false; }
	if( i >= _m ) { return false; }
	if( j >= _n ) { return false; }

	const int& r0 = r[i];
	const int& r1 = r[i+1];

	if( i > j ) { // searching in the lower part

		for( int k=r0; k<r1; ++k )
		{
			if( c[k] == j )
			{
				return true; // (i,j)-entry exists!
			}
		}

	} else { // searching in the upper part

		for( int k=r1-1; k>=r0; --k )
		{
			if( c[k] == j )
			{
				return true; // (i,j)-entry exists!
			}
		}

	}

	return false; // (i,j)-entry doesn't exist!
}

template <class T>
bool
ZSparseMatrix<T>::isNonZero( const int& i, const int& j ) const
{
	const int& r0 = r[i];
	const int& r1 = r[i+1];

	if( i > j ) { // searching in the lower part

		for( int k=r0; k<r1; ++k )
		{
			if( c[k] == j )
			{
				return true; // (i,j)-entry exists!
			}
		}

	} else { // searching in the upper part

		for( int k=r1-1; k>=r0; --k )
		{
			if( c[k] == j )
			{
				return true; // (i,j)-entry exists!
			}
		}

	}

	return false; // (i,j)-entry doesn't exist!
}

template <class T>
bool
ZSparseMatrix<T>::isSymmetric( float tolerance ) const
{
	int k = 0;
	for( int i=0; i<_nnz; ++i )
	{
		if( r[k] <= i ) // only for 
		{
			const int& row    = k;
			const int& column = c[i];

			const T& a = (*this)( row, column );
			const T& b = (*this)( column, row );

			if( !ZAlmostSame( a, b, tolerance ) )
			{
				return false;
			}

			if( (i+1) == r[k+1] )
			{
				++k; // increase row-index when it reaches the end of row-j.
			}
		}
	}

	return true;
}

template <class T>
bool
ZSparseMatrix<T>::isBlockSymmetric( float tolerance ) const
{
	int k = 0;
	for( int i=0; i<_nnz; ++i )
	{
		if( r[k] <= i ) // only for 
		{
			const int& row    = k;
			const int& column = c[i];

			const T& a = (*this)(row,column);
			const T& b = (*this)(column,row);

			if( !a.isAlmostSame( b, tolerance ) )
			{
				return false;
			}

			if( (i+1) == r[k+1] )
			{
				++k; // increase row-index when it reaches the end of row-j.
			}
		}
	}

	return true;
}

template <class T>
inline void
ZSparseMatrix<T>::print() const
{
	const int m = (*this).m();
	const int n = (*this).n();

	// Matrix information.
	cout << "===================" << endl;
	cout << "( " << m << " X " << n << " ) Matrix" << endl;
	cout << "# of non-zeros = " << (*this).nnz() << endl;
	cout << "v.size() = " << (*this).v.size() << endl;
	cout << "r.size() = " << (*this).r.size() << endl;
	cout << "c.size() = " << (*this).c.size() << endl;
	cout << "===================" << endl;

	FOR( i, 0, m )
	{
		cout << "[ ";
		FOR( j, 0, n )
		{
			int value = 0;
			(*this).getValue( i, j, value );
			cout << value << " ";
		}
		cout << "]" << endl;
	}
	cout << endl;
}

template <class T>
inline void
ZSparseMatrix<T>::write( ofstream& fout ) const
{
	fout.write( (char*)&_m,   sizeof(int) );
	fout.write( (char*)&_n,   sizeof(int) );
	fout.write( (char*)&_nnz, sizeof(int) );

	fout.write( (char*)&v[0], v.size()*sizeof(T)   );
	fout.write( (char*)&r[0], r.size()*sizeof(int) );
	fout.write( (char*)&c[0], c.size()*sizeof(int) );
}

template <class T>
inline void
ZSparseMatrix<T>::read( ifstream& fin )
{
	fin.read( (char*)&_m,   sizeof(int) );
	fin.read( (char*)&_n,   sizeof(int) );
	fin.read( (char*)&_nnz, sizeof(int) );

	set( _m, _n, _nnz );

	fin.read( (char*)&v[0], (_nnz)*sizeof(T)   );
	fin.read( (char*)&r[0], (_m+1)*sizeof(int) );
	fin.read( (char*)&c[0], (_nnz)*sizeof(int) );
}

template <class T>
inline bool
ZSparseMatrix<T>::save( const char* filePathName ) const
{
	ofstream fout( filePathName, ios::out|ios::binary|ios::trunc );

	if( fout.fail() || !fout.is_open() )
	{
		cout << "Error@ZSparseMatrix(): Failed to save file: " << filePathName << endl;
		return false;
	}

	write( fout );

	fout.close();

	return true;
}

template <class T>
inline bool
ZSparseMatrix<T>::load( const char* filePathName )
{
	ifstream fin( filePathName, ios::in|ios::binary );

	if( fin.fail() )
	{
		cout << "Error@ZSparseMatrix(): Failed to load file." << endl;
		reset();
		return false;
	}

	read( fin );

	fin.close();

	return true;
}

/////////////////////////////////////////
// simple non-member utility funcitons //
/////////////////////////////////////////

template <class T>
inline ostream&
operator<<( ostream& os, const ZSparseMatrix<T>& m )
{
	return os;
}

////////////////
// data types //
////////////////

typedef ZSparseMatrix<int>		ZSparseMatrixi;
typedef ZSparseMatrix<float>	ZSparseMatrixf;
typedef ZSparseMatrix<double>	ZSparseMatrixd;

ZELOS_NAMESPACE_END

#endif

