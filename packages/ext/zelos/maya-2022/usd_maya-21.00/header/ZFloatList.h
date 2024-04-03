//--------------//
// ZFloatList.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZFloatList_h_
#define _ZFloatList_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZFloatList : public ZList<float>
{
	private:

		typedef ZList<float> parent;

	public:

		ZFloatList();
		ZFloatList( const ZFloatList& l );

		float min() const;
		float max() const;
		float absMax() const;
};

ostream&
operator<<( ostream& os, const ZFloatList& object );

ZELOS_NAMESPACE_END

#endif

