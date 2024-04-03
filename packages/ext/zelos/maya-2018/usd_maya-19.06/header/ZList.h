//---------//
// ZList.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZList_h_
#define _ZList_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief Linked list.
/**
	This class provides a linked list.
	Common convenience functions are available, and the implementation is compatible with the internal Zelos implementation so that it can be passed efficiently between internal Zelos data streuctures.
*/
template <class T>
class ZList : public std::list<T>
{
	private:

		typedef std::list<T> parent;

	public:

		ZList();
		ZList( const ZList<T>& l );

		void reset();

		int length() const;

		T& append( const T& element );
		void append( const ZList<T>& list );

		void exchange( ZList<T>& other );

		ZList<T>& operator=( const ZList<T>& other );

		ZString dataType() const;
};

template <class T>
ZList<T>::ZList()
{}

template <class T>
ZList<T>::ZList( const ZList<T>& l )
: std::list<T>::list()
{
	parent::assign( l.begin(), l.end() );
}

template <class T>
void ZList<T>::reset()
{
	parent::clear();
}

template <class T>
int ZList<T>::length() const
{
	return (int)parent::size();
}

template <class T>
T& ZList<T>::append( const T& e )
{
	parent::push_back( e );
	return *parent::rbegin();
}

template <class T>
void ZList<T>::append( const ZList<T>& l )
{
	typename std::list<T>::const_iterator itr = l.begin();
	for( ; itr!=l.end(); ++itr )
	{
		parent::push_back( *itr );
	}
}

template <class T>
void ZList<T>::exchange( ZList<T>& l )
{
	(*this).swap( l );
}

template <class T>
ZList<T>& ZList<T>::operator=( const ZList<T>& l )
{
	parent::assign( l.begin(), l.end() );
	return (*this);
}

template <class T>
ZString ZList<T>::dataType() const
{
	ZString type( "ZList_" );
	return ( type + typeid(T).name() );
}

template <class T>
ostream&
operator<<( ostream& os, const ZList<T>& object )
{
	os << "<ZList>" << endl;
	os << " data type: " << object.dataType() << endl;
	os << " size     : " << object.size() << endl;
	os << " memory   : " << ZString::commify(object.size()*sizeof(T)) << " bytes" << endl;
	os << endl;

//	typename std::object<T>::const_iterator itr = object.begin();
//	for( int i=0; itr!=object.end(); ++itr, ++i )
//	{
//		os << i << ": "<< *itr << endl;
//	}

	return os;
}

ZELOS_NAMESPACE_END

#endif

