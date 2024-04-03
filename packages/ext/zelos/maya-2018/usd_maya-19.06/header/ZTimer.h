//----------//
// ZTimer.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZTimer_h_
#define _ZTimer_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// @brief Timer to check performance.
/**
	This class provides the functionality for calculating the time elapsed/consumed in executing a particular portion of the code.
	The portion of the code to be timed is embedded inbetween the calls to 'start()' and 'stop()'.
	A call to 'check()' gives the time spent in executing that portion of the code.
*/
class ZTimer
{
	protected:

		#ifdef OS_LINUX
			struct timeval _start;
			struct timeval _end;
			struct timeval _elapsed;
		#endif

		#ifdef OS_WINDOWS
			LARGE_INTEGER _start;
			LARGE_INTEGER _end;
			LARGE_INTEGER _elapsed;
			LARGE_INTEGER _freq;
		#endif

	public:

		ZTimer();
		ZTimer( const ZTimer& t );

		ZTimer& operator=( const ZTimer& other );

		void start();
		float check( bool asMilliSecond=true );
		float stop( bool asMilliSecond=true );
};

ostream&
operator<<( ostream& os, const ZTimer& object );

ZELOS_NAMESPACE_END

#endif

