//------------//
// ZGlUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.05.25                               //
//-------------------------------------------------------//

#ifndef _ZGlUtils_h_
#define _ZGlUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/////////////
// glColor //

inline void
glColorBlack()
{ glColor3f( 0.0f, 0.0f, 0.0f ); }

inline void
glColorWhite()
{ glColor3f( 1.0f, 1.0f, 1.0f ); }

inline void
glColorRed()
{ glColor3f( 1.0f, 0.0f, 0.0f ); }

inline void
glColorGreen()
{ glColor3f( 0.0f, 1.0f, 0.0f ); }

inline void
glColorBlue()
{ glColor3f( 0.0f, 0.0f, 1.0f ); }

inline void
glColorYellow()
{ glColor3f( 1.0f, 1.0f, 0.0f ); }

inline void
glColorMagenta()
{ glColor3f( 1.0f, 0.0f, 1.0f ); }

inline void
glColorCyan()
{ glColor3f( 0.0f, 1.0f, 1.0f ); }

inline void
glColorOrange()
{ glColor3f( 1.0f, 0.6f, 0.0f ); }

inline void
glColorGray()
{ glColor3f( 0.5f, 0.5f, 0.5f ); }

inline void
glColor( const float& c )
{ glColor3f( c, c, c ); }

inline void
glColor( const float& c, const float& a )
{ glColor4f( c, c, c, a ); }

inline void glColor( const float& r, const float& g, const float& b )
{ glColor3f( r, g, b ); }

inline void glColor( const double& r, const double& g, const double& b )
{ glColor3f( r, g, b ); }

inline void glColor( const float& r, const float& g, const float& b, const float& a )
{ glColor4f( r, g, b, a ); }

inline void glColor( const double& r, const double& g, const double& b, const double& a )
{ glColor4d( r, g, b, a ); }

inline void glColor( const ZPoint& c )
{ glColor3fv( &c.x ); }

inline void glColor( const ZColor& c )
{ glColor4fv( &c.r ); }

inline void glColor( const ZVector& c, const float& a )
{ glColor4f( c.x, c.y, c.z, a ); }

inline void glColor( const ZColor& c, const float& a )
{ glColor4f( c.r, c.g, c.b, a ); }

//////////////
// glVertex //

inline void
glVertex( const int& x, const int& y )
{ glVertex2i( x, y ); }

inline void
glVertex( const float& x, const float& y )
{ glVertex2f( x, y ); }

inline void
glVertex( const double& x, const double& y )
{ glVertex2d( x, y ); }

inline void
glVertex( const int& x, const int& y, const int& z )
{ glVertex3i( x, y, z ); }

inline void
glVertex( const float& x, const float& y, const float& z )
{ glVertex3f( x, y, z ); }

inline void
glVertex( const double& x, const double& y, const double& z )
{ glVertex3d( x, y, z ); }

inline void
glVertex( const ZPoint& p )
{ glVertex3fv( &p.x ); }

inline void
glVertex( const ZFloat2& p )
{ glVertex3f( p[0], p[1], 0.f ); }

//////////////
// glNormal //

inline void
glNormal( const float& x, const float& y, const float& z )
{ glNormal3f( x, y, z ); }

inline void
glNormal( const double& x, const double& y, const double& z )
{ glNormal3d( x, y, z ); }

inline void
glNormal( const ZVector& n )
{ glNormal3fv( &n.x ); }

////////////
// glLine //

inline void
glLine( const ZPoint& p, const ZVector& v )
{ glVertex(p); glVertex3f(p.x+v.x,p.y+v.y,p.z+v.z); }

///////////////////////////
// matrix multiplication //

inline void glMultMatrix( const float* M )
{ glMultMatrixf(M); }

inline void glMultMatrix( const double* M )
{ glMultMatrixd(M); }

void ZMultiplyMatrixByVector( float dst[4], float mat[16], float src[3] );

/////////////////
// translation //

inline void glTranslate( const float& tx, const float& ty, const float& tz )
{ glTranslatef( tx, ty, tz ); }

inline void glTranslate( const double& tx, const double& ty, const double& tz )
{ glTranslated( tx, ty, tz ); }

inline void glTranslate( const ZVector& t )
{ glTranslatef( t.x, t.y, t.z ); }

//////////////
// rotation //

inline void glRotate( const float& angle, const float& rx, const float& ry, const float& rz )
{ glRotatef( angle, rx, ry, rz ); }

inline void glRotate( const double& angle, const double& rx, const double& ry, const double& rz )
{ glRotated( angle, rx, ry, rz ); }

inline void glRotate( const ZVector& r )
{
	glRotatef( r.z, 0.f, 0.f, 1.f );
	glRotatef( r.y, 0.f, 1.f, 0.f );
	glRotatef( r.x, 1.f, 0.f, 0.f );
}

///////////
// scale //

inline void glScale( const float& s )
{ glScalef( s, s, s ); }

inline void glScale( const float& sx, const float& sy, const float& sz )
{ glScalef( sx, sy, sz ); }

inline void glScale( const double& sx, const double& sy, const double& sz )
{ glScaled( sx, sy, sz ); }

inline void glScale( const ZVector& s )
{ glScalef( s.x, s.y, s.z ); }

void DrawFullScreenQuad();

void ZResetDisplayList( int& dispListId );

void ZCallDisplayList( const int& dispListId );

void ZDrawCube( const ZPoint& minPt, const ZPoint& maxPt );
void ZDrawCircle( float radius, int numSlices, int numLoops );
void ZDrawCircleOnXYPlane( const ZPoint& center, float radius, int numSegments, bool asWireFrame );
void ZDrawCircleOnYZPlane( const ZPoint& center, float radius, int numSegments, bool asWireFrame );
void ZDrawCircleOnZXPlane( const ZPoint& center, float radius, int numSegments, bool asWireFrame );
void ZDrawArcLineOnXYPlane( const ZPoint& center, float radius, float startAngle, float endAngle, int numSegs );
void ZDrawArcLineOnYZPlane( const ZPoint& center, float radius, float startAngle, float endAngle, int numSegs );
void ZDrawArcLineOnZXPlane( const ZPoint& center, float radius, float startAngle, float endAngle, int numSegs );
void ZDrawSphere( const float& radius, const int& numSlices, const int& numStacks );

ZPoint ZCurrentCameraPosition();

ZELOS_NAMESPACE_END

#endif

