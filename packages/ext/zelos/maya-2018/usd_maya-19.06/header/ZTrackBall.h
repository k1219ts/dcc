//--------------//
// ZTrackBall.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZTrackBall_h_
#define _ZTrackBall_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// @brief Virtual trackBall rotation controller
class ZTrackBall
{
	private:

		float       _w;			// width  of window
		float       _h;			// height of window
		ZPoint      _c;			// center of the sphere (=virutal trackball)
		float       _r;			// radius of the sphere (=virtual trackball)
		bool        _dragging;	// flag: true=dragging, false=nothing
		ZPoint      _pCurrent;	// current mouse position
		ZPoint      _pClicked;	// mouse position at the beginning of dragging
		ZQuaternion _qCurrent;	// current rotation
		ZQuaternion _qReleased;	// rotation after the dragging
		ZMatrix     _mCurrent;	// current rotation matrix

	public:

		ZTrackBall();
		void setWindowSize( float width, float height );
		void setCurrentPosition( int x, int y );
		void start();
		void finish();
        const ZMatrix& rotationMatrix() const;

	private:

		// Calculate rotation matrix from two positions (beginning/current).
		void _update();

		// map: mouse position -> point on the sphere
		ZPoint _map( const ZPoint& mouse, const ZPoint& center, float r ) const;
};

inline
ZTrackBall::ZTrackBall()
{
	_w = _h = 0;
	_r = 1.f;
	_dragging = false;
}

inline void
ZTrackBall::setWindowSize( float width, float height )
{
	_w = width;
	_h = height;
}

inline void
ZTrackBall::setCurrentPosition( int x, int y )
{
	_pCurrent.set( 2*(x/float(_w))-1, 2*((_h-y)/float(_h))-1, 0.f );
	_update();
}

inline void
ZTrackBall::start()
{
	_dragging = true;
	_pClicked = _pCurrent;
}

inline void
ZTrackBall::finish()
{
	_dragging = false;
	_qReleased = _qCurrent;
}

inline const ZMatrix&
ZTrackBall::rotationMatrix() const
{
	return _mCurrent;
}

inline void
ZTrackBall::_update()
{
	const ZPoint from( _map( _pClicked, _c, _r ) );
	const ZPoint to  ( _map( _pCurrent, _c, _r ) );
	if( _dragging ) { _qCurrent.setArcRotation(from,to); _qCurrent*=_qReleased; }
	_qCurrent.toRotationMatrix( _mCurrent );
}

inline ZPoint
ZTrackBall::_map( const ZPoint& mouse, const ZPoint& center, float r ) const
{
	ZPoint p( (mouse-center) );
	p *= 1 / r;
	const float l2 = p.squaredLength();
	if(l2>1) { p.normalize(); p.z=0.f; }
	else     { p.z = sqrtf(1-l2);      }
	return p;
}

ZELOS_NAMESPACE_END

#endif

