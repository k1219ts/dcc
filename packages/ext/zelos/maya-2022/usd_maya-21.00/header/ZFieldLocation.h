//------------------//
// ZFieldLocation.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.26                               //
//-------------------------------------------------------//

#ifndef _ZFieldLocation_h_
#define _ZFieldLocation_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZFieldLocation
{
	public:

		//enum FieldLocation
		typedef enum
		{
			zNone = 0, ///< none
			zCell = 1, ///< cell center
			zNode = 2, ///< node
			zFace = 3  ///< face center
		} FieldLocation;

	public:

		ZFieldLocation() {}

		static ZString name( ZFieldLocation::FieldLocation definedLocation )
		{
			switch( definedLocation )
			{
				default:
				case ZFieldLocation::zNone: { return ZString("none"); }
				case ZFieldLocation::zCell: { return ZString("cell"); }
				case ZFieldLocation::zNode: { return ZString("node"); }
				case ZFieldLocation::zFace: { return ZString("face"); }
			}
		}
	
		ZFieldLocation::FieldLocation value( int idx )
		{
			switch( idx )
			{
				default:
				case 0: { return ZFieldLocation::zNone; }
				case 1: { return ZFieldLocation::zCell; }
				case 2: { return ZFieldLocation::zNode; }
				case 3: { return ZFieldLocation::zFace; }			
			}
		}
};

inline ostream&
operator<<( ostream& os, const ZFieldLocation& object )
{
	os << "<ZFieldLocation>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

