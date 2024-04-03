//--------------//
// ZTestClass.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.02.16                               //
//-------------------------------------------------------//

#ifndef _ZTestClass_h_
#define _ZTestClass_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZTestClass
{
	public:

	public:

		ZTestClass();
		ZTestClass( const ZTestClass& object );

		std::vector<double> return_double_tuple();
		std::vector<float> return_float_tuple();

		std::vector<double> times2( std::vector<double> a );
};

ostream&
operator<<( ostream& os, const ZTestClass& object );

ZELOS_NAMESPACE_END

#endif

