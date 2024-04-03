//--------//
// ZPtc.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
//         Nayoung Kim @ Dexter Studios                  //
// last update: 2016.09.20                               //
//-------------------------------------------------------//

#ifndef _ZPtc_h_
#define _ZPtc_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

enum PTC_TYPE 
{
	PTC_NONE=0,
	PTC_FOAM,
	PTC_BUBBLE,
	PTC_SPLASH,
	PTC_SPRAY,
	PTC_DEL
};

class ZPtc
{
	public:

		ZString      name; // name
		int          lIdx; // last index
		int          gUid; // group id
		ZColor       gClr; // group color
		ZBoundingBox aabb; // axis-aligned bounding box
		float        tScl; // time scale

		ZIntArray    uid; // unique id
		ZPointArray  pos; // position
		ZVectorArray vel; // velocity
		ZFloatArray  rad; // radius
		ZColorArray  clr; // color
		ZVectorArray nrm; // normal
		ZVectorArray vrt; // vorticity
		ZFloatArray  dst; // density
		ZFloatArray  sdt; // signed distance
		ZPointArray  uvw; // texture coordinate
		ZFloatArray  age; // age
		ZFloatArray  lfs; // lifespan 
        ZIntArray    sts; // status
		ZIntArray    typ; // type

	public:

		ZPtc();

		void reset();

		int count() const;
		int numAttributes() const;

		ZPtc& operator=( const ZPtc& other );

		void remove( const ZIntArray& delList );
		int  deadList( ZIntArray& list );

		double usedMemorySize( ZDataUnit::DataUnit dataUnit=ZDataUnit::zBytes ) const;

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
		
		bool savePtc( const char* filename) const;
		bool loadPtc( const char* filename);

		void drawPos( bool useGroupColor ) const;
        void drawVel( const float maxVel );
};

ostream&
operator<<( ostream& os, const ZPtc& ptc );

ZELOS_NAMESPACE_END

#endif

