//----------------//
// ZParticleSet.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.06.01                               //
//-------------------------------------------------------//

#ifndef _ZParticleSet_h_
#define _ZParticleSet_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZParticleSet
{
	public:

	private:

		int _attrBitMask=0;

	public:

		ZParticleSet();
};

ostream&
operator<<( ostream& os, const ZParticleSet& ptc );

ZELOS_NAMESPACE_END

#endif

