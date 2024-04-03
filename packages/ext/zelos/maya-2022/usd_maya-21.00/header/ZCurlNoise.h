//--------------//
// ZCurlNoise.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jinhyuk Bae @ Dexter Studios                  //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZCurlNoise_h_
#define _ZCurlNoise_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZCurlNoise : public ZSimplexNoise
{
	private:

		float _delta;
		float _denom;

	public:

		ZCurlNoise();
	
		ZVector velocity( float x, float y, float z, float t );
};

ostream&
operator<<( ostream& os, const ZCurlNoise& object );

ZELOS_NAMESPACE_END

#endif

