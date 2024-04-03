#ifndef _BS_GLDrawer_h_
#define _BS_GLDrawer_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

static void Draw( const ScreenMesh& mesh, const VectorArray* vertexColorsPtr=nullptr )
{
    const int n = mesh.numTriangles();
    if( n == 0 ) { return; }

    const VectorArray& p = mesh.p;
    const UIntArray&   t = mesh.t;

    if( vertexColorsPtr )
    {
        const VectorArray& c = *vertexColorsPtr;

        glBegin( GL_TRIANGLES );
        {
            for( int i=0; i<n; ++i )
            {
                const int index = 3*i;

                const int& v0 = t[index  ];
                const int& v1 = t[index+1];
                const int& v2 = t[index+2];

                glColor( c[v0] );   glVertex( p[v0] );
                glColor( c[v1] );   glVertex( p[v1] );
                glColor( c[v2] );   glVertex( p[v2] );
            }
        }
        glEnd();
    }
    else
    {
        glBegin( GL_TRIANGLES );
        {
            for( int i=0; i<n; ++i )
            {
                const int index = 3*i;

                const int& vrt0 = t[index  ];
                const int& vrt1 = t[index+1];
                const int& vrt2 = t[index+2];

                glVertex( p[vrt0] );
                glVertex( p[vrt1] );
                glVertex( p[vrt2] );
            }
        }
        glEnd();
    }
}

static void DrawUV( const ScreenMesh& mesh )
{
    const int n = mesh.numTriangles();
    if( n == 0 ) { return; }

    const UIntArray&    t = mesh.t;
    const VectorArray& uv = mesh.uv;

    glBegin( GL_TRIANGLES );
    {
        for( int i=0; i<n; ++i )
        {
            const int index = 3*i;

            const int& vrt0 = t[index  ];
            const int& vrt1 = t[index+1];
            const int& vrt2 = t[index+2];

            glVertex2d( uv[vrt0].x, uv[vrt0].y );
            glVertex2d( uv[vrt1].x, uv[vrt1].y );
            glVertex2d( uv[vrt2].x, uv[vrt2].y );
        }
    }
    glEnd();
}

BS_NAMESPACE_END

#endif

