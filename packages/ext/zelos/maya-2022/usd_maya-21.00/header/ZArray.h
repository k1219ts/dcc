//----------//
// ZArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2019.01.16                               //
//-------------------------------------------------------//

#ifndef _ZArray_h_
#define _ZArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A 1D array class.
/**
	This class implements an array of various data types in Zelos system.
	Common convenience functions are available, and the implementation is compatible with the internal Zelos implementation so that it can be passed efficiently between internal Zelos data streuctures.
	It is inherited from "STL vector" class.
	Therefore, all functions of "STL vector" such as push_back(), empty(), reserve(), etc. are also available.
	@warning
	The "class T" of which size is not fix may cause some problems especially when sizeof(T) function is used.
	Therefore, "class T" is recommended not to have any virtual functions.
	If not, be careful to use the functions to use 'sizeof(T)' inside its member functions. \n
	ex) zeroize(), save(), load(), ...
*/
template <class T>
class ZArray : public std::vector<T>
{
	private:

		typedef std::vector<T> parent;

	public:

		/// @brief The default constructor.
		/**
			It creates a new empty instance which contains no elements.
		*/
		ZArray();

		/// @brief The copy constructor.
		/**
			It creates a new array instance and initializes the instance to the same contents as the given array.
			@param[in] source The array object to copy from.
		*/
		ZArray( const ZArray<T>& source );

		/// @brief The copy constructor.
		/**
			It creates a new array instance and initializes the instance to the same contents as the given list.
			@param[in] source The list object to copy from.
		*/
		ZArray( const ZList<T>& source );

		/// @brief The copy constructor.
		/**
			It creates a new array instance so that the instance has the given length and every element has zero bits.
			@param[in] initialLength The initial length of the array.
		*/
		ZArray( int initialLength );

		/// @brief The class constructor.
		/**
			It creates a new array instance so that the instance has the given length and each element has the given value.
			@param[in] initialLength The initial length of the array.
			@param[in] valueForAll The initial value for all the elements.
		*/
		ZArray( int initialLength, const T& valueForAll );

		/// @brief The class constructor.
		/**
			It creates a new array instance and initializes the instance to the same contents as the cache file by the given name.
			If this function meets some errors while loading from the file, the instance will be reset.
			@param[in] filePathName The file path and name to load from it.
		*/ 
		ZArray( const char* filePathName );

		/// @brief The function to reset the instance.
		/**
			It clears all the contents of the array so that the array instance has the zero length.
 		*/
		void reset();

		/// @brief The function to set the instance.
		/**
			It re-initializes the instance by the given arguments.
			This function will grow and shrink the array as desired.
			Elements that are grown have uninitialized values, while those which are shrunk will lose the data contained in the deleted elements (i.e. it will release the memory).
			@param[in] length The length of the array.
			@param[in] initializeAsZeros If true, it initialize all its bits to zero.
		*/
		void setLength( int length, bool initializeAsZeros=true );

		/// @brief The function to set the instance.
		/**
			It re-initializes the instance by the given arguments.
			This function will grow and shrink the array as desired.
			Elements that are grown have uninitialized values, while those which are shrunk will lose the data contained in the deleted elements (ie. it will release the memory).
			@param[in] length The length of the array.
			@param[in] valueForAll The value for all the elements.
		*/
		void setLengthWithValue( int length, const T& valueForAll );

		/// @brief The index operator.
		/**
			It returns the reference of the element at the given index.
			@note The valid indices are 0 to length()-1. No range checking is done for efficiency.
			@param[in] index The index of the element whose value is to be returned as reference.
			@return The reference of the indicated element.
		*/
		T& operator()( const int& index );

		/// @brief The index operator.
		/**
			It returns the value of the element at the given index.
			@note The valid indices are 0 to length()-1. No range checking is done for efficiency.
			@param[in] index The index of the element whose value is to be returned.
			@return The value of the indicated element.
		*/
		const T& operator()( const int& index ) const;

		/// @brief The reference of the first element.
		/**
			It returns the reference of the first element.
			@return The reference of the first element.
		*/
		T& first();

		/// @brief The value of the first element.
		/**
			It returns the value of the first element.
			@return The value of the first element.
		*/
		const T& first() const;

		/// @brief The reference of the last element.
		/**
			It returns the reference of the last element.
			@return The reference of the last element.
		*/
		T& last();

		/// @brief The value of the last element.
		/**
			It returns the value of the last element.
			@return The value of the last element.
		*/
		const T& last() const;

		/// @brief The function for creating a new reduces array.
		/**
			It creates a new array instance and copies from the given array and indices mask.
			@param[in] mask The flags for indicating the elements to be copied from.
		*/
		void from( const ZArray<T>& other, const ZArray<char>& mask );

		/// @brief The assignement operator.
		/**
			It copies all of the elements of the other array instance into this one.
			@param[in] other The array being copied from.
			@return The reference of this array instance.
		*/
		ZArray<T>& operator=( const ZArray<T>& other );

		/// @brief The assignement operator.
		/**
			It copies all of the elements of the other list instance into this one.
			@param[in] other The list being copied from.
			@return The reference of this array instance.
		*/
		ZArray<T>& operator=( const std::list<T>& other );

		/// @brief The equality operator.
		/**
			It returns true if this array has the same elements and order as the given array.
			@param[in] other The array to be compared.
			@return True if same, and false otherwise.
		*/
		bool operator==( const ZArray<T>& other ) const;

		/// @brief The inequality operator.
		/**
			It returns true if this array does not have the same elements and order as the given array.
			@param[in] other The array to be compared.
			@return True if different, and false otherwise.
		*/
		bool operator!=( const ZArray<T>& other ) const;

		/// @brief The in-place addition operator.
		/**
			Each element of this array instance are added by the given value.
			@note The in-place addition operator must be defined at the class T.
			@param[in] value The value being added to each element.
			@return The reference of this array instance.
		*/
		ZArray<T>& operator+=( T value );

		/// @brief The in-place subtraction operator.
		/**
			Each element of this array instance are subtracted by the given value.
			@note The in-place subtraction operator must be defined at the class T.
			@param[in] value The value being subtracted from each element.
			@return The reference of this array instance.
		*/
		ZArray<T>& operator-=( T value );

		/// @brief The length of the array.
		/**
			It returns the number of elements in the instance.
			@return The number of elements in the instance.
		*/
		int length() const;

		T* pointer( const int& startIndex=0 );

		const T* pointer( const int& startIndex=0 ) const;

		/// @brief The function for appending an element.
		/**
			It adds a new element to the end of the array.
			@note Since it calls std::vector::push_back(), push_back() is faster than append().
			However, it is different from push_back() in that it returns the reference of the element to be just added.
			@param[in] element The value for the new last element being added.
			@return The reference of the newly added last element.
		*/
		T& append( const T& element );

		/// @brief The function for appending elements.
		/**
			It adds new elements to the end of the array.
			@param[in] elements The array being added at the end.
		*/
		void append( const ZArray<T>& elements );

		/// @brief The function for appending elements.
		/**
			It adds new elements to the end of the array.
			@param[in] elements The list being added at the end.
		*/
		void append( const ZList<T>& elements );

		/// @brief The function for sorting the elements into ascending order.
		/**
			It sorts the elements into ascending order.
		*/
		void sort();

		/// @brief The function for reversing the order of the elements.
		/**
			It reverses the order of the elements.
		*/
		void reverse();

        int findTheFirstIndex( const T& value ) const;

		/// @brief The function for removing elements.
		/**
			It removes the array elements at the given indices.
			All array elements following the removed element are shifted toward the first element.
			@param[in] indicesToBeDeleted The array of indices to be deleted.
			@return The number of elements in the array after removed.
		*/
		int remove( const ZArray<int>& indicesToBeDeleted );

		/// @brief The function for removing elements.
		/**
			It removes the array elements at the given indices.
			All array elements following the removed element are shifted toward the first element.
			@param[in] indicesToBeDeleted The list of indices to be deleted.
			@return The number of elements in the array after removed.
		*/
		int remove( const ZIntList& indicesToBeDeleted );

		/// @brief The function for removing repeated elements.
		/**
			It finds the elements appeared consecutively with a same value, and removes them all-but-one.
		*/
		void eliminateRepeatedElements();

		/// @brief The function for removing duplicated elements.
		/**
			It eliminates all the redundant elements in the array.
		*/
		void deduplicate();

		/// @brief The function for removing duplicated elements.
		/**
			It eliminates all the redundant elements in the array, and sorts it.
		*/
		void deduplicateAndSort();

		/// @brief The function for making every elements zero.
		/**
			It set the block of the memory to zero so that the every element of the array instance has zero bits.
		*/
		void zeroize();

		/**
			It set all the elements as the given value.
			@param[in] valueForAll The value for all the elements.
		*/
		void fill( const T& valueForAll );

		void inverse();

		/// @brief The function for mixing elements.
		/**
			It mixes up all elements of the array randomly.
		*/
		void shuffle( int seed=0 );

		/// @brief The function for swapping data between two arrays.
		/**
			It exchanges data between two arrays.
			It exchanges the contents of the array by the given contents of the array, which is another array of the same type, but sizes may differ.
			@note The global function std::swap() have the same behavior.
			@param[in] other The array of the same type whose contents being swapped with that of the array.
		*/
		void exchange( ZArray<T>& other );

		void split( const ZArray<int>& groupId, vector<ZArray<T> >& result ) const;

		const ZString dataType() const;

		void write( ofstream& fout, bool writeNumElements=false ) const;
		void write( gzFile& gzf, bool writeNumElements=false ) const;

		void read( ifstream& fin, bool readNumElements=false );
		void read( gzFile& gzf, bool readNumElements=false );

		/// @brief The function for saving the array.
		/**
			It saves the data of the array into a file.
			@param[in] filePathName The file path and name to save the data into it.
			@return True if success, and false otherwise.
		*/
		bool save( const char* filePathName ) const;

		/// @brief The function for loading the array.
		/**
			It loads the data of the array from a file.
			@param[in] filePathName The file path and name to load the data from it.
			@return True if success, and false otherwise.
		*/
		bool load( const char* filePathName );

		/// @brief The size of allocated memory size.
		/**
			It returns the allocated memory size.
			@note It may differ from the size of the cache file saved by save() function \n
			because the cache file can have the length of the array as an integer value.
			@param[in] dataUnit The unit of the data size being queried.
			@return The allocated memory size.
		*/
		double usedMemorySize( ZDataUnit::DataUnit dataUnit=ZDataUnit::zBytes ) const;

		void checkIndex( int index ) const;

		void print( bool horizontal=true, int maxIndex=-1 ) const;
};

template <class T>
ZArray<T>::ZArray()
{}

template <class T>
ZArray<T>::ZArray( const ZArray<T>& a )
: std::vector<T>::vector()
{
	parent::assign( a.begin(), a.end() );
}

template <class T>
ZArray<T>::ZArray( const ZList<T>& l )
: std::vector<T>::vector()
{
	parent::assign( l.begin(),l.end() );
}

template <class T>
ZArray<T>::ZArray( int initialLength )
: std::vector<T>::vector()
{
	parent::resize( initialLength );
	ZArray<T>::zeroize();
}

template <class T>
ZArray<T>::ZArray( int initialLength, const T& valueForAll )
: std::vector<T>::vector()
{
	parent::resize( initialLength );
	ZArray<T>::fill( valueForAll );
}

template <class T>
ZArray<T>::ZArray( const char* filePathName )
: std::vector<T>::vector()
{
	ZArray<T>::load( filePathName );
}

template <class T>
inline void
ZArray<T>::reset()
{
	if( parent::empty() ) { return; }
	parent::clear();
}

template <class T>
inline void
ZArray<T>::setLength( int length, bool initializeAsZeros )
{
	if( length<=0 ) { parent::clear(); return; }
	if( (int)parent::size() != length ) { parent::resize(length); }
	if( initializeAsZeros ) { ZArray<T>::zeroize(); }
}

template <class T>
inline void
ZArray<T>::setLengthWithValue( int length, const T& valueForAll )
{
	if( length<=0 ) { parent::clear(); return; }
	if( (int)parent::size() != length ) { parent::resize(length); }
	ZArray<T>::fill( valueForAll );
}

template <class T>
inline const T&
ZArray<T>::operator()( const int& i ) const
{
	return parent::operator[](i);
}

template <class T>
inline T&
ZArray<T>::operator()( const int& i )
{
	return parent::operator[](i);
}

template <class T>
inline const T&
ZArray<T>::first() const
{
	return parent::front();
}

template <class T>
inline T&
ZArray<T>::first()
{
	return parent::front();
}

template <class T>
inline const T&
ZArray<T>::last() const
{
	return parent::back();
}

template <class T>
inline T&
ZArray<T>::last()
{
	return parent::back();
}

template <class T>
void
ZArray<T>::from( const ZArray<T>& other, const ZArray<char>& mask )
{
	parent::clear();

    if( other.length() == 0 ) { return; }

	const int nElements = other.length();

	if( nElements != mask.length() )
	{
		cout << "Error@ZArray::from(): Invalid input data." << endl;
		return;
	}

	int count = 0;
	{
		FOR( i, 0, nElements )
		{
			if( mask[i] )
			{
				++count;
			}
		}
	}

	parent::reserve( count );

	FOR( i, 0, nElements )
	{
		if( mask[i] )
		{
			parent::push_back( other[i] );
		}
	}
}

template <class T>
inline ZArray<T>&
ZArray<T>::operator=( const ZArray<T>& a )
{
	parent::assign( a.begin(), a.end() );
	return (*this);
}

template <class T>
inline ZArray<T>&
ZArray<T>::operator=( const std::list<T>& a )
{
	parent::assign( a.begin(), a.end() );
	return (*this);
}

template <class T>
inline bool
ZArray<T>::operator==( const ZArray<T>& a ) const
{
	const int n = (int)parent::size();

	FOR( i, 0, n )
	{
		if( parent::operator[](i) != a[i] )
		{
			return false;
		}
	}

	return true;
}

template <class T>
inline bool
ZArray<T>::operator!=( const ZArray<T>& a ) const
{
	const int n = (int)parent::size();

	FOR( i, 0, n )
	{
		if( parent::operator[](i) != a[i] )
		{
			return true;
		}
	}

	return false;
}

template <class T>
inline ZArray<T>&
ZArray<T>::operator+=( T v )
{
	const int n = (int)parent::size();

	FOR( i, 0, n )
	{
		parent::operator[](i) += v;
	}

	return (*this);
}

template <class T>
inline ZArray<T>&
ZArray<T>::operator-=( T v )
{
	const int n = (int)parent::size();

	FOR( i, 0, n )
	{
		parent::operator[](i) -= v;
	}

	return (*this);
}

template <class T>
inline int
ZArray<T>::length() const
{
	return (int)parent::size();
}

template <class T>
inline T*
ZArray<T>::pointer( const int& i )
{
	if( parent::empty() ) { return (T*)NULL; }
	return (T*)(&parent::operator[](i));
}

template <class T>
inline const T*
ZArray<T>::pointer( const int& i ) const
{
	if( parent::empty() ) { return (T*)NULL; }
	return (T*)(&parent::operator[](i));
}

template <class T>
inline T&
ZArray<T>::append( const T& e )
{
	parent::push_back( e );
	return parent::back();
}

template <class T>
inline void
ZArray<T>::append( const ZArray<T>& a )
{
	parent::resize( parent::size()+a.size() );
	std::copy( a.begin(), a.end(), parent::end()-a.size() );
}

template <class T>
inline void
ZArray<T>::append( const ZList<T>& l )
{
	if( l.empty() ) { return; }
	parent::resize( parent::size()+l.size() );
	std::copy( l.begin(), l.end(), parent::end()-l.size() );
}

template <class T>
void
ZArray<T>::sort()
{
	std::sort( parent::begin(), parent::end() );
}

template <class T>
void
ZArray<T>::reverse()
{
	std::reverse( parent::begin(), parent::end() );
}

template <class T>
int
ZArray<T>::findTheFirstIndex( const T& value ) const
{
	const int n = (int)parent::size();
    FOR( i, 0, n ) { if( parent::at(i) == value ) { return i; } }
    return -1;
}

template <class T>
int
ZArray<T>::remove( const ZArray<int>& indicesToBeDeleted )
{
	const int n = (int)parent::size();
	if( !n ) { return 0; }

	const int listSize = (int)indicesToBeDeleted.size();
	if( !listSize ) { return n; }

	std::vector<bool> mask( n, false );

	int numToDelete = 0;
	std::vector<int>::const_iterator itr = indicesToBeDeleted.begin();
	for( ; itr!=indicesToBeDeleted.end(); ++itr )
	{
		const int& idx = *itr;
		if( idx <  0  ) { continue; }
		if( idx >= n  ) { continue; }
		if( mask[idx] ) { continue; } // already checked
		mask[idx] = true;
		++numToDelete;
	}

	if( numToDelete == n ) { parent::clear(); return 0; }

	const int finalSize = n - numToDelete;

	std::vector<T> tmp( finalSize );

	for( int i=0, count=0; i<n; ++i )
	{
		if( mask[i] ) { continue; }
		tmp[count++] = parent::operator[](i);
	}

	parent::swap( tmp );

	return finalSize;
}

template <class T>
int
ZArray<T>::remove( const ZIntList& indicesToBeDeleted )
{
	const int n = (int)parent::size();
	if( !n ) { return 0; }

	const int listSize = (int)indicesToBeDeleted.size();
	if( !listSize ) { return n; }

	std::vector<bool> mask( n, false );

	int numToDelete = 0;
	std::list<int>::const_iterator itr = indicesToBeDeleted.begin();
	for( ; itr!=indicesToBeDeleted.end(); ++itr )
	{
		if( *itr >= n ) { continue; }
		if( mask[*itr] ) { continue; } // already checked
		mask[*itr] = true;
		++numToDelete;
	}

	if( numToDelete == n ) { parent::clear(); return 0; }

	const int finalSize = n - numToDelete;

	std::vector<T> tmp( finalSize );

	for( int i=0, count=0; i<n; ++i )
	{
		if( mask[i] ) { continue; }
		tmp[count++] = parent::operator[](i);
	}

	parent::swap( tmp );

	return finalSize;
}

template <class T>
inline void
ZArray<T>::eliminateRepeatedElements()
{
	std::unique( parent::begin(), parent::end() );
}

template <class T>
void
ZArray<T>::deduplicate()
{
	const int n = (int)parent::size();

	ZArray<T> tmp;
	parent::swap( tmp );
	parent::clear();

	std::vector<char> alreadyExist( n, (char)0 );

	FOR( i, 0, n )
	{
		if( alreadyExist[i] ) { continue; }

		FOR( j, i+1, n )
		{
			if( tmp[i] == tmp[j] ) { alreadyExist[j] = (char)1; }
		}
	}

	FOR( i, 0, n )
	{
		if( alreadyExist[i] ) { continue; }
		parent::push_back( tmp[i] );
	}
}

template <class T>
inline void
ZArray<T>::deduplicateAndSort()
{
	std::set<T> tmp( parent::begin(), parent::end() );
	parent::assign( tmp.begin(), tmp.end() );
}

template <class T>
inline void
ZArray<T>::zeroize()
{
	if( parent::empty() ) { return; }
	memset( (char*)&parent::operator[](0), 0, parent::size()*sizeof(T) );
}

template <class T>
inline void
ZArray<T>::fill( const T& valueForAll )
{
	if( parent::empty() ) { return; }
	if( ZIsZero(valueForAll) ) { ZArray<T>::zeroize(); return; }
	std::fill( parent::begin(), parent::end(), valueForAll );
}

template <class T>
inline void
ZArray<T>::inverse()
{
	const int n = (int)parent::size();

	FOR( i, 0, n )
	{
		T& a = parent::operator[](i);
		a = !a;
	}
}

template <class T>
inline void
ZArray<T>::shuffle( int seed )
{
	std::srand( seed );
	std::random_shuffle( parent::begin(), parent::end(), ZRandInt0 );
}

template <class T>
inline void
ZArray<T>::exchange( ZArray<T>& a )
{
	parent::swap( a );
}

template <class T>
void
ZArray<T>::split( const ZArray<int>& groupId, vector<ZArray<T> >& result ) const
{
	result.clear();

	const int n = (int)parent::size();
	if( !n ) { return; }

	if( n != groupId.length() )
	{
		cout << "Error@ZArray::split(): Invalid input data." << endl;
		return;
	}

	int maxGroupId = 0;

	FOR( i, 0, n )
	{
		if( groupId[i] < 0 )
		{
			cout << "Error@ZArray::split(): Invalid input data." << endl;
			return;
		}

		maxGroupId = ZMax( maxGroupId, groupId[i] );
	}	

	result.resize( maxGroupId );

	FOR( i, 0, n )
	{
		result[ groupId[i] ].push_back( (char*)&parent::operator[](i) );
	}
}

template <class T>
inline const ZString
ZArray<T>::dataType() const
{
	ZString type( "ZArray_" );
	return ( type + typeid(T).name() );
}

template <class T>
inline void
ZArray<T>::write( ofstream& fout, bool writeNumElements ) const
{
	const int n = (int)parent::size();
	if( writeNumElements ) { fout.write( (char*)&n, sizeof(int) ); }
	if( n ) { fout.write( (char*)&parent::operator[](0), n*sizeof(T) ); }
}

template <class T>
inline void
ZArray<T>::write( gzFile& gzf, bool writeNumElements ) const
{
	const int n = (int)parent::size();
	if( writeNumElements ) { gzwrite( gzf, (char*)&n, sizeof(int) ); }
	if( n ) { gzwrite( gzf, (char*)&parent::operator[](0), n*sizeof(T) ); }
}

template <class T>
inline void
ZArray<T>::read( ifstream& fin, bool readNumElements )
{
	int n = (int)parent::size();
	if( readNumElements ) { fin.read( (char*)&n, sizeof(int) ); parent::resize(n); }
	if( n ) { fin.read( (char*)&parent::operator[](0), n*sizeof(T) ); }
	else { parent::clear(); }
}

template <class T>
inline void
ZArray<T>::read( gzFile& gzf, bool readNumElements )
{
	int n = (int)parent::size();
	if( readNumElements ) { gzread( gzf, (char*)&n, sizeof(int) ); parent::resize(n); }
	if( n ) { gzread( gzf, (char*)&parent::operator[](0), n*sizeof(T) ); }
	else { parent::clear(); }
}

template <class T>
inline bool
ZArray<T>::save( const char* filePathName ) const
{
	ofstream fout( filePathName, ios::out|ios::binary|ios::trunc );

	if( fout.fail() || !fout.is_open() )
	{
		cout << "Error@ZArray::save(): Failed to save file: " << filePathName << endl;
		return false;
	}

	dataType().write( fout, true );
	ZArray<T>::write( fout, true );

	fout.close();

	return true;
}

template <class T>
inline bool
ZArray<T>::load( const char* filePathName )
{
	ifstream fin( filePathName, ios::in|ios::binary );

	if( fin.fail() )
	{
		cout << "Error@ZArray::load(): Failed to load file." << endl;
		reset();
		return false;
	}

	ZString type;
	type.read( fin, true );
	if( type != dataType() )
	{
		cout << "Error@ZArray::load(): Data type mismatch." << endl;
		reset();
		return false;
	}

	ZArray<T>::read( fin, true );

	fin.close();

	return true;
}

template <class T>
inline double
ZArray<T>::usedMemorySize( ZDataUnit::DataUnit dataUnit ) const
{
	const double bytes = parent::size() * sizeof(T);

	switch( dataUnit )
	{
		case ZDataUnit::zBytes:     { return bytes;                 }
		case ZDataUnit::zKilobytes: { return (bytes/(1024.0));      }
		case ZDataUnit::zMegabytes: { return (bytes/ZPow2(1024.0)); }
		case ZDataUnit::zGigabytes: { return (bytes/ZPow3(1024.0)); }
		default: { cout<<"Error@ZArray::usedMemorySize(): Invalid data unit." << endl; return 0; }
	}
}

template <class T>
inline void
ZArray<T>::checkIndex( int idx ) const
{
	if( idx<0 || idx>=(int)parent::size() )
	{
		cout << "Error@Array: Invalid index." << endl;
		exit(0);
	}
}

template <class T>
inline void
ZArray<T>::print( bool horizontal, int maxIndex ) const
{
	if( maxIndex < 0 ) {

		if( horizontal ) {

			std::copy( parent::begin(), parent::end(), std::ostream_iterator<T>( cout, " " ) );
			cout << endl;

		} else { // vertical

			std::copy( parent::begin(), parent::end(), std::ostream_iterator<T>( cout, "\n" ) );
			cout << endl;

		}

	} else {

		const int len = length();
		const int n = (maxIndex<0) ? len : ZMin( len, maxIndex );

		if( horizontal ) {

			FOR( i, 0, n )
			{
				cout << parent::operator[](i) << " ";
			}
			cout << endl;

		} else { // vertical

			FOR( i, 0, n )
			{
				cout << parent::operator[](i) << endl;
			}

		}

	}
}

template <class T>
inline ostream&
operator<<( ostream& os, const ZArray<T>& object )
{
	os << "<ZArray>" << endl;
	os << " data type: " << object.dataType() << endl;
	os << " size     : " << object.size() << " (" << (object.size()*sizeof(T)/ZPow2(1024.0)) << " mb)" << endl;
	os << " capacity : " << object.capacity() << " (" << (object.capacity()*sizeof(T)/ZPow2(1024.0)) << " mb)" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

