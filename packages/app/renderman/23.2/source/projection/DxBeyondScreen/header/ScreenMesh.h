#ifndef _BS_ScreenMesh_h_
#define _BS_ScreenMesh_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class ScreenMesh
{
    public:

        VectorArray p;  // vertex positions in object space
        UIntArray   t;  // triangle connections
        VectorArray uv; // vertex positions in uv space

        // note) p.length() = uv.length()

    public:

        ScreenMesh()
        {
            // nothing to do
        }

        void clear()
        {
            p .clear();
            t .clear();
            uv.clear();
        }

        int numVertices() const
        {
            return (int)p.size();
        }

        int numTriangles() const
        {
            return (int)(t.size()/3);
        }

        ScreenMesh& operator=( const ScreenMesh& m )
        {
            p  = m.p;
            t  = m.t;
            uv = m.uv;

            return (*this);
        }

        Vector center() const
        {
            return p.center();
        }

        BoundingBox boundingBox() const
        {
            return p.boundingBox();
        }

        void transform( const Matrix& m )
        {
            const int n = ScreenMesh::numVertices();

            #pragma omp parallel for
            for( int i=0; i<n; ++i )
            {
                p[i] = m.transform( p[i], false );
            }
        }

        Vector intersectionPoint( const Vector& P, const Vector& Q ) const
        {
            const int n = numTriangles();

            VectorArray points( n );
            DoubleArray dists( n );

            dists.fill( INFINITE );

            #pragma omp parallel for
            for( int i=0; i<n; ++i )
            {
                const int index = 3*i;

                const int& v0 = t[index  ];
                const int& v1 = t[index+1];
                const int& v2 = t[index+2];

                const Vector& A = p[v0];
                const Vector& B = p[v1];
                const Vector& C = p[v2];

                Double3 baryCoords;
                double t = 0.0;

                if( RayTriangleTest( P,Q-P, A,B,C, baryCoords, t, 1e-3 ) )
                {
                    Vector& R = points[i];

                    R.x = ( baryCoords[0] * A.x ) + ( baryCoords[1] * B.x ) + ( baryCoords[2] * C.x );
                    R.y = ( baryCoords[0] * A.y ) + ( baryCoords[1] * B.y ) + ( baryCoords[2] * C.y );
                    R.z = ( baryCoords[0] * A.z ) + ( baryCoords[1] * B.z ) + ( baryCoords[2] * C.z );

                    dists[i] = P.distanceTo( R );
                }
            }

            double min_dist = INFINITE;
            Vector intersection_point;

            for( int i=0; i<n; ++i )
            {
                if( dists[i] < min_dist )
                {
                    min_dist = dists[i];
                    intersection_point = points[i];
                }
            }

            return intersection_point;
        }

        void write( ofstream& fout ) const
        {
            p .write( fout );
            t .write( fout );
            uv.write( fout );
        }

        void read( ifstream& fin )
        {
            p .read( fin );
            t .read( fin );
            uv.read( fin );
        }

        bool save( const char* filePathName ) const
        {
            ofstream fout( filePathName, ios::out|ios::binary|ios::trunc );

            if( fout.fail() || !fout.is_open() )
            {
                cout << "Error@ScreenMesh::save(): Failed to save file: " << filePathName << endl;
                return false;
            }

            ScreenMesh::write( fout );

            fout.close();

            return true;
        }

        bool load( const char* filePathName )
        {
            ScreenMesh::clear();

            ifstream fin( filePathName, ios::in|ios::binary );

            if( fin.fail() )
            {
                cout << "Error@ZArray::load(): Failed to load file." << endl;
                return false;
            }

            ScreenMesh::read( fin );

            fin.close();

            return true;
        }
};

BS_NAMESPACE_END

#endif

