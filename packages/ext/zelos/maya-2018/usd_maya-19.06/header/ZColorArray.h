//---------------//
// ZColorArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.15                               //
//-------------------------------------------------------//

#ifndef _ZColorArray_h_
#define _ZColorArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZColorArray : public ZArray<ZColor>
{
	private:

		typedef ZArray<ZColor> parent;

		ZColorSpace::ColorSpace _colorSpace;

	public:

		ZColorArray();
		ZColorArray( const ZColorArray& a );
		ZColorArray( int initialLength );
		ZColorArray( int initialLength, const ZColor& valueForAll );

//		void add( const float& r, const float& g, const float& b, const float& a );

		ZColorSpace::ColorSpace colorSpace() const;
		void setColorSpace( ZColorSpace::ColorSpace colorSpace );

		ZColorArray& operator*=( float scalar );

		void rgb2hsv();
		void hsv2rgb();

		void setRandomColors( int seed=0, float min=0.2f, float max=1.1f );

		const ZString dataType() const;

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

//inline void
//ZColorArray::add( const float& r, const float& g, const float& b, const float& a )
//{
//	std::vector<ZColor>::emplace_back( r, g, b, a );
//}

ostream&
operator<<( ostream& os, const ZColorArray& object );

ZELOS_NAMESPACE_END

#endif

