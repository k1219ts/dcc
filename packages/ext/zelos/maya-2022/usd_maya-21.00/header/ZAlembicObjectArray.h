//-----------------------//
// ZAlembicObjectArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.01.20                               //
//-------------------------------------------------------//

#ifndef _ZAlembicObjectArray_h_
#define _ZAlembicObjectArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZAlembicObjectArray
{
	private:

		vector<ZAlembicObject*> _data;

	public:

		ZAlembicObjectArray();

		~ZAlembicObjectArray();

		void reset();

		void append( ZAlembicObject* elementPtr );

		void append( const ZAlembicObject& element );

		int length() const;

		ZAlembicObject& operator[]( const int& i );
		ZAlembicObject& operator()( const int& i );
		ZAlembicObject& last( const int& i );

		const ZAlembicObject& operator[]( const int& i ) const;
		const ZAlembicObject& operator()( const int& i ) const;
		const ZAlembicObject& last( const int& i ) const;

		void reverse();

		int remove( const ZIntArray& indicesToBeDeleted );

		void getFrameRange( double& minFrame, double& maxFrame ) const;

		void getTimeRange( double& minTime, double& maxTime ) const;
};

ostream&
operator<<( ostream& os, const ZAlembicObjectArray& object );

ZELOS_NAMESPACE_END

#endif

