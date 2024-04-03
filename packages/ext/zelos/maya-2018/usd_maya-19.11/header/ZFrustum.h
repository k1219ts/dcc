//------------//
// ZFrustum.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.29                               //
//-------------------------------------------------------//

#ifndef _ZFrustum_h_
#define _ZFrustum_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A 3D view frustum of a camera.
/**
	A 3D volume that is visible from a camera.
	This class treats a symmetrical, quadrilateral, and perspective view frustum only
	, which has the shape of a frustum of a pyramid with the top part cut off.
*/
// The view frustum of a camera is represented by six planes.
// : near, far, left, right, top, and bottom
class ZFrustum
{
	private:

		ZDirection::Direction _initialAim;       // the initial aiming direction

		float   _near;             // the distance from the view-point(=camera position) to view-port(=image plane)
		float   _far;              // the maximum distance bounding visibilities of objects in space
		float   _horizontalFOV;    // the horizontal field of view
		float   _verticalFOV;      // the vertical field of view
		float   _width;            // the width of the image plane
		float   _height;           // the height of the image plane
		float   _aspectRatio;      // = width / height

		ZPoint  _eyePosition;      // the eye position in world space
		ZMatrix _xform;            // the 4x4 transformation matrix

		ZPoint  _localCorners[8];  // the eight corner points in local space
		ZPoint  _worldCorners[8];  // the eight corner points in world space

		ZPlane  _planes[6];        // to be built from world corner points

		static const int triangleIndices[12][3];

	public:

		ZFrustum();
		ZFrustum( const ZFrustum& frustum );

		void reset();

		ZFrustum& set( float near, float far, float horizontalFOV, float aspectRatio, ZDirection::Direction initialAim=ZDirection::zNegative );
		ZFrustum& set( float cs, float lsr, float ar, float hfa, float vfa, float ncp, float fcp, float fl, bool o, float ow, ZDirection::Direction initialAim=ZDirection::zNegative );

		ZFrustum& operator=( const ZFrustum& frustum );

		void applyTransform( const ZMatrix& xform );

		const ZPlane& nearPlane() const;
		const ZPlane& farPlane() const;
		const ZPlane& leftPlane() const;
		const ZPlane& rightPlane() const;
		const ZPlane& topPlane() const;
		const ZPlane& bottomPlane() const;

		bool contains( const ZPoint& point ) const;

		bool contains( const ZSphere& sphere ) const;
		bool intersects( const ZSphere& sphere ) const;

		bool contains( const ZBoundingBox& aabb ) const;
		bool intersects( const ZBoundingBox& aabb ) const;

		void draw() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );
};

ostream& operator<<( ostream& os, const ZFrustum& object );

ZELOS_NAMESPACE_END

#endif

