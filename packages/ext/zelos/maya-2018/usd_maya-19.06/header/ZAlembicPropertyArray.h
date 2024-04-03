//-------------------------//
// ZAlembicPropertyArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.01.20                               //
//-------------------------------------------------------//

#ifndef _ZAlembicPropertyArray_h_
#define _ZAlembicPropertyArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZAlembicPropertyArray
{
	private:

		vector<ZAlembicProperty*> _data;

	public:

		ZAlembicPropertyArray();

		~ZAlembicPropertyArray();

		void reset();

		void append( ZAlembicProperty* elementPtr );

		void append( const ZAlembicProperty& element );

		int length() const;

		ZAlembicProperty& operator[]( const int& i );
		ZAlembicProperty& operator()( const int& i );
		ZAlembicProperty& last( const int& i );

		const ZAlembicProperty& operator[]( const int& i ) const;
		const ZAlembicProperty& operator()( const int& i ) const;
		const ZAlembicProperty& last( const int& i ) const;

		void reverse();
};

ostream&
operator<<( ostream& os, const ZAlembicPropertyArray& object );

ZELOS_NAMESPACE_END

#endif

