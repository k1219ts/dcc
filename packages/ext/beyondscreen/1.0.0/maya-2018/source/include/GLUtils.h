#ifndef _BS_GLUtils_h_
#define _BS_GLUtils_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

inline void glVertex( const double& x, const double& y, const double& z ) { glVertex3d( x, y, z ); }
inline void glVertex( const Vector& v ) { glVertex3d( v.x, v.y, v.z ); }

inline void glColor( const double& x, const double& y, const double& z ) { glColor3d( x, y, z ); }
inline void glColor( const Vector& v ) { glColor3d( v.x, v.y, v.z ); }

BS_NAMESPACE_END

#endif

