//-------------//
// ZGlCamera.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZGlCamera_h_
#define _ZGlCamera_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZGlCamera
{
	private:

		enum Mode { IDLE, PANNING, ZOOMING };

		Mode       _mode;
		float      _winWidth;
		float      _winHeight;
		float      _zoomSpeed;			// a zoomming factor
		ZTrackBall _trackBall;
    	ZMatrix    _viewMatrix;
    	ZVector    _eyeDirection;		// from eye to center (store this vector rather than the eye position)
    	ZPoint     _center;				// the point at which the camera looks
    	ZPoint     _mousePosLast;
		ZPoint     _mousePosCurrent;

    public:

		ZGlCamera();
		ZGlCamera( const ZPoint& eye, const ZPoint& center );

		ZPoint getEye() const;

		void setCenter( const ZPoint& center );
		void setEye( const ZPoint& eye );

		void setCurrentMousePos( float x, float y );
		void setWindowSize( float w, float h );

		void beginRotate();
		void endRotate();

		void beginPan();
		void endPan();

		void beginZoom();
		void endZoom();

		void pan( float deltaX, float deltaY );
		void zoom( float deltaZ );

		void getViewMatrix( float viewMatrix[16] ) const;

	private:

		void _init();
		void _updateViewMatrix();
};

ZELOS_NAMESPACE_END

#endif

