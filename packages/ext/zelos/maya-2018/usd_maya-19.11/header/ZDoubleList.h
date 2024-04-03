//---------------//
// ZDoubleList.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZDoubleList_h_
#define _ZDoubleList_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDoubleList : public ZList<double>
{
	private:

		typedef ZList<double> parent;

	public:

		ZDoubleList();
		ZDoubleList( const ZDoubleList& l );

		double min() const;
		double max() const;
		double absMax() const;
};

ostream&
operator<<( ostream& os, const ZDoubleList& object );

ZELOS_NAMESPACE_END

#endif

