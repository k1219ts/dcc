//---------//
// ZHeap.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZHeap_h_
#define _ZHeap_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

template <class T, class S>
struct ZHeapNode
{
	T data;
	S value;

	ZHeapNode( const T& inData, const S& inValue )
	: data(inData), value(inValue)
	{}
};

template <class T, class S>
class ZMinHeap
{
	protected:

		struct Compare
		{
			bool operator()( const ZHeapNode<T,S>& a, const ZHeapNode<T,S>& b )
			{
				return ( a.value > b.value );
			}
		};

		priority_queue< class ZHeapNode<T,S>, vector<class ZHeapNode<T,S> >, class ZMinHeap<T,S>::Compare > _data;

	public:

		ZMinHeap() {}

		void push( const ZHeapNode<T,S>& n ) { _data.push(n); }

		const class ZHeapNode<T,S>& top() { return _data.top(); }

		void pop() { return _data.pop(); }

		bool empty() { return _data.empty(); }

		int size() const { return (int)_data.size(); }

		void clear() { _data = priority_queue< class ZHeapNode<T,S>, vector<class ZHeapNode<T,S> >, class ZMinHeap<T,S>::Compare >(); }
};

template <class T, class S>
class ZMaxHeap
{
	protected:

		struct Compare
		{
			bool operator()( const ZHeapNode<T,S>& a, const ZHeapNode<T,S>& b )
			{
				return ( a.value < b.value );
			}
		};

		priority_queue< class ZHeapNode<T,S>, vector<class ZHeapNode<T,S> >, class ZMaxHeap<T,S>::Compare > _data;

	public:

		ZMaxHeap() {}

		void push( const ZHeapNode<T,S>& n ) { _data.push(n); }

		const class ZHeapNode<T,S>& top() { return _data.top(); }

		void pop() { return _data.pop(); }

		bool empty() { return _data.empty(); }

		int size() const { return (int)_data.size(); }

		void clear() { _data = priority_queue< class ZHeapNode<T,S>, vector<class ZHeapNode<T,S> >, class ZMaxHeap<T,S>::Compare >(); }
};

////////////////
// data types //

typedef ZHeapNode<ZInt3,float> ZHEAPNODE;

ZELOS_NAMESPACE_END

#endif

