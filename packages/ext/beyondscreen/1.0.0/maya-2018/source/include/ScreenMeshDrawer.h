#ifndef _BS_ScreenMeshDrawer_h_
#define _BS_ScreenMeshDrawer_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class ScreenMeshDrawer
{
    private:

        enum VertexBuffer
        {
            POS_BUFFER, // vertex positions in world space
            UVW_BUFFER, // vertex positions in uvw space
            IDX_BUFFER, // triangle indices

            NUM_VERTEX_BUFFERS // # of vertex buffers
        };

    private:

        size_t _numPoints   = 0;
        size_t _numVertices = 0;

        GLProgram glProgram; // OpenGL shader program

        GLuint vertexArrayId = 0;                  // VAO
        GLuint vertexBufferId[NUM_VERTEX_BUFFERS]; // VBOs

        int hasFisheyeTexture = 0;
        GLuint fisheyeTextureId = 0;

    public:

        ScreenMeshDrawer()
        {
            // nothing to do
        }

        ~ScreenMeshDrawer()
        {
            ScreenMeshDrawer::reset();
        }

        void reset()
        {
            if( vertexArrayId > 0 )
            {
                glDeleteVertexArrays( 1, &vertexArrayId );
                vertexArrayId = 0;

                glDeleteBuffers( NUM_VERTEX_BUFFERS, vertexBufferId );
            }

            if( fisheyeTextureId > 0 )
            {
                glDeleteTextures( 1, &fisheyeTextureId );
            }
        }

        void draw( const Manager& manager, const Image& fisheyeImage )
        {
            const ScreenMesh& worldScreenMesh     = manager.worldScreenMesh;
            const Vector&     worldAimingPoint    = manager.worldAimingPoint;
            const Vector&     worldCameraPosition = manager.worldCameraPosition;
            const Vector&     worldCameraUpvector = manager.worldCameraUpvector;

            if( glProgram.id() == 0 )
            {
                glProgram.addShader( GL_VERTEX_SHADER,   ScreenMeshVS );
                glProgram.addShader( GL_FRAGMENT_SHADER, ScreenMeshFS );
                glProgram.link();
            }

            const VectorArray& POS = worldScreenMesh.p;
            const VectorArray& UVW = worldScreenMesh.uv;
            const UIntArray&   TRI = worldScreenMesh.t;

            const int numPoints   = worldScreenMesh.numVertices();
            const int numVertices = worldScreenMesh.numTriangles() * 3;

            if( ( vertexArrayId == 0 ) || ( _numPoints != numPoints ) || ( _numVertices != numVertices ) )
            {
                ScreenMeshDrawer::reset();

                glGenVertexArrays( 1, &vertexArrayId );

                glBindVertexArray( vertexArrayId );
                {
                    glGenBuffers( NUM_VERTEX_BUFFERS, vertexBufferId );

                    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, vertexBufferId[IDX_BUFFER] );
                    glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*numVertices, &(TRI[0]), GL_STATIC_DRAW );
                }
                glBindVertexArray( 0 );

                _numPoints   = numPoints;
                _numVertices = numVertices;
            }

            if( vertexArrayId > 0 )
            {
                glProgram.enable();
                {
                    glProgram.bindModelViewMatrix( "modelViewMatrix"   );
                    glProgram.bindProjectionMatrix( "projectionMatrix" );

                    glProgram.bind( "worldAimingPoint", worldAimingPoint );
                    glProgram.bind( "worldCameraPosition", worldCameraPosition );
                    glProgram.bind( "worldCameraUpvector", worldCameraUpvector );

                    hasFisheyeTexture = ScreenMeshDrawer::setTexture( fisheyeImage );
                    glProgram.bind( "hasFisheyeTexture", hasFisheyeTexture );
                    glProgram.bindTexture( "fisheyeTexture", fisheyeTextureId, GL_TEXTURE_2D );

                    glBindVertexArray( vertexArrayId );
                    {
                        glBindBuffer( GL_ARRAY_BUFFER, vertexBufferId[POS_BUFFER] );
                        glBufferData( GL_ARRAY_BUFFER, sizeof(Vector)*numPoints, &(POS[0]), GL_DYNAMIC_DRAW );
                        glEnableVertexAttribArray( POS_BUFFER );
                        glVertexAttribPointer( POS_BUFFER, 3, GL_DOUBLE, GL_FALSE, 0, 0 );

                        glBindBuffer( GL_ARRAY_BUFFER, vertexBufferId[UVW_BUFFER] );
                        glBufferData( GL_ARRAY_BUFFER, sizeof(Vector)*numPoints, &(UVW[0]), GL_DYNAMIC_DRAW );
                        glEnableVertexAttribArray( UVW_BUFFER );
                        glVertexAttribPointer( UVW_BUFFER, 3, GL_DOUBLE, GL_FALSE, 0, 0 );

                        glDrawElements( GL_TRIANGLES, numVertices, GL_UNSIGNED_INT, 0 );
                    }
                    glBindVertexArray( 0 );

                    glBindTexture( GL_TEXTURE_2D, 0 );
                }
                glProgram.disable();
            }
        }

    private:

        bool setTexture( const Image& fisheyeImage )
        {
            if( fisheyeImage.numPixels() == 0 )
            {
                return false;
            }

            if( fisheyeTextureId > 0 )
            {
                glDeleteTextures( 1, &fisheyeTextureId );
            }

            glGenTextures( 1, &fisheyeTextureId );

            const Image& img = fisheyeImage;
            const Pixel* imgData = img.pointer();

            glBindTexture( GL_TEXTURE_2D, fisheyeTextureId );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0, GL_RGBA, GL_HALF_FLOAT, imgData );
            glBindTexture( GL_TEXTURE_2D, 0 );

            return true;
        }
};

BS_NAMESPACE_END

#endif

