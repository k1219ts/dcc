//------------//
// ZIntList.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZIntList_h_
#define _ZIntList_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZIntList : public ZList<int>
{
	private:

		typedef ZList<int> parent;

	public:

		ZIntList();
		ZIntList( const ZIntList& l );

		int min() const;
		int max() const;
		int absMax() const;
};

ostream&
operator<<( ostream& os, const ZIntList& object );

ZELOS_NAMESPACE_END

#endif

