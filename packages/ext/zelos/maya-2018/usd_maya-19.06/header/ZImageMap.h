//-------------//
// ZImageMap.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.11.01                               //
//-------------------------------------------------------//

#ifndef _ZImageMap_h_
#define _ZImageMap_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZImageMap
{
	protected:

		int                       _width;			// image width
		int                       _height;			// image height
		int                       _numChannels;		// number of channels
		float*                    _data;			// image data
		ZImageFormat::ImageFormat _format;			// image format
		ZString                   _filePathName;	// image file name

		static float _255; // 1/255

	public:

		ZImageMap();
		ZImageMap( const ZImageMap& img );
		ZImageMap( const char* filePathName );
		ZImageMap( int Width, int Height, int NumChannels );

		virtual ~ZImageMap();

		void reset();

		bool set( int Width, int Height, int NumChannels );

		ZImageMap& operator=( const ZImageMap& img );

		int width() const                        { return _width;                        }
		int height() const                       { return _height;                       }
		int numChannels() const                  { return _numChannels;                  }
		int size() const                         { return (_width*_height*_numChannels); }
		ZImageFormat::ImageFormat format() const { return _format;                       }
		ZString filePathName() const             { return _filePathName;                 }

		// i: width, j: height, k: channel
		int index( const int& i, const int& j, const int& k ) const;

		const float& operator[]( const int& idx ) const;
		float& operator[]( const int& idx );

		const float& operator()( const int& i, const int& j, const int& k ) const;
		float& operator()( const int& i, const int& j, const int& k );

		void setPixelColor( const int& i, const int& j, const ZColor& color, bool setAlpha ) const;

		ZColor color( const int& i, const int& j, bool getAlpha ) const;

		float intensity( const int& i, const int& j ) const;
		float intensity( const ZPoint& pos ) const;

		// u,v = [0,1]
		float lerp( const float& u, const float& v, const int k=0 ) const;
		float fastValue( const float& u, const float& v, int k=0 ) const;
		ZColor fastColor( const float& u, const float& v ) const;

        float min( const int& channelIndex ) const;
        float max( const int& channelIndex ) const;

        float average( const int& channelsIndex ) const;

        void histogram( ZIntArray& counts, const int& numSegments, const int& channelIndex, const float& min=0.f, const float& max=1.f ) const;

		float* pointer() { return _data; }
		const float* pointer() const { return _data; }

		bool save( const char* filePathName, ZImageFormat::ImageFormat imageFormat ) const;
		bool load( const char* filePathName, bool skipIfSameFile=false );

	protected:

		void _init();

		bool _load_tif( const char* filePathName );
		//bool _load_jpg( const char* filePathName );
		bool _load_tga( const char* filePathName );
};

inline int
ZImageMap::index( const int& i, const int& j, const int& k ) const
{
	return (k+_numChannels*(i+j*_width));
}

inline const float&
ZImageMap::operator[]( const int& idx ) const
{
	return _data[idx];
}

inline float&
ZImageMap::operator[]( const int& idx )
{
	return _data[idx];
}

inline const float&
ZImageMap::operator()( const int& i, const int& j, const int& k ) const
{
	return _data[k+_numChannels*(i+j*_width)];
}

inline float&
ZImageMap::operator()( const int& i, const int& j, const int& k )
{
	return _data[k+_numChannels*(i+j*_width)];
}

inline void
ZImageMap::setPixelColor( const int& i, const int& j, const ZColor& color, bool setAlpha ) const
{
	int idx = _numChannels*(i+j*_width);

	_data[  idx] = color.r;
	_data[++idx] = color.g;
	_data[++idx] = color.b;
	if( setAlpha && _numChannels>3 ) { _data[++idx] = color.a; }
}

inline ZColor
ZImageMap::color( const int& i, const int& j, bool getAlpha ) const
{
	int idx = _numChannels*(i+j*_width);
	ZColor c;
	c.r = (float)_data[  idx];
	c.g = (float)_data[++idx];
	c.b = (float)_data[++idx];
	if( getAlpha ) { c.a = (float)_data[++idx]; }
	return c;
}

inline float
ZImageMap::intensity( const int& i, const int& j ) const
{
	int idx = _numChannels*(i+j*_width);
	float sum = 0.299f * _data[idx];
	sum += 0.587f * _data[++idx];
	sum += 0.114f * _data[++idx];
	return sum;
}

inline float
ZImageMap::intensity( const ZPoint& pos ) const
{
	const float x = pos.x;
	const float y = pos.z;
	int i=int(x); float fx=x-i;
	int j=int(y); float fy=y-j;

	if(i<0) {i=0;fx=0;} else if(i>=_width-1 ) {i=_width-2; fx=1;}
	if(j<0) {j=0;fy=0;} else if(j>=_height-1) {j=_height-2;fy=1;}

	int idx[4];
	idx[0] = _numChannels*(i  +_width*j    ); 
	idx[1] = _numChannels*(i+1+_width*j    ); 
	idx[2] = _numChannels*(i+1+_width*(j+1)); 
	idx[3] = _numChannels*(i  +_width*(j+1));
	
	const float _fx=1-fx, _fy=1-fy;
	const float wgt[4] = { _fx*_fy, fx*_fy, fx*fy, _fx*fy };
	float est = (float)wgt[0] * (0.299*_data[idx[0]]+0.587*_data[idx[0]+1]+0.114*_data[idx[0]+2]);
	FOR(l,1,4) { est += (float)wgt[l]*(0.299*_data[idx[l]]+0.587*_data[idx[l]+1]+0.114*_data[idx[l]+2]); }
	
	return est;
}

inline float
ZImageMap::fastValue( const float& u, const float& v, int k ) const
{
	const int i = ZClamp( int(    u*_width ), 0, _width-1  );
	const int j = ZClamp( int((1-v)*_height), 0, _height-1 );
	const int idx = _numChannels*(i+j*_width);
	return (float)_data[idx+k];
}

inline float
ZImageMap::lerp( const float& u, const float& v, const int k ) const
{
	float px = float(_width-1)*u;
	float py = float(_height-1)*v;

	const int i = ZMin( int(px), _width-2 );
	const int j = ZMin( int(py), _height-2 );
	const int idx = i+j*_width;
	const int c = _numChannels;

	px = px-float(i);
	py = py-float(j);

	float v00 = _data[idx*c+k]*(1.f-px) + _data[(idx+1)*c+k]*px;
	float v11 = _data[(idx+_width)*c+k]*(1.f-px) + _data[(idx+_width+1)*c+k]*px;

	return v00*(1.f-py) + v11*py;
}

inline ZColor
ZImageMap::fastColor( const float& u, const float& v ) const
{
	const int i = ZClamp( int(    u*_width ), 0, _width-1  );
	const int j = ZClamp( int((1-v)*_height), 0, _height-1 );
	int idx = _numChannels*(i+j*_width);
	ZColor c;
	c.r = (float)_data[  idx];
	c.g = (float)_data[++idx];
	c.b = (float)_data[++idx];
	return c;
}

#define PER_EACH_PIXEL( image )             \
	for( int i=0; i<image.width();  ++i ) { \
	for( int j=0; j<image.height(); ++j ) {

#define END_PER_EACH_PIXEL }}

ostream&
operator<<( ostream& os, const ZImageMap& object );

ZELOS_NAMESPACE_END

#endif

