#ifndef _BS_Manager_h_
#define _BS_Manager_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class Manager
{
    public:

        // for the i-th animation frame
        // worldScreenMesh = objectToWorldMatrices[i] x objectScreenMesh

        // cache data
        IntArray    animationFrames;
        ScreenMesh  objectScreenMesh;
        Int4        cornerIndices;
        MatrixArray objectToWorldMatrices;
        VectorArray worldAimingPoints;
        VectorArray worldCameraPositions;
        VectorArray worldCameraUpvectors;

        // derived data
        Matrix      objectToWorldMatrix;
        ScreenMesh  worldScreenMesh;
        BoundingBox worldScreenMeshAABB;
        Vector      worldAimingPoint;
        Vector      worldCameraPosition;
        Vector      worldCameraUpvector;

        // the hash grid of the screen mesh in uv space
        HashGrid2D  hashGrid;

    public:

        Manager()
        {
            // nothing to do
        }

        void clear()
        {
            animationFrames       .clear();
            objectScreenMesh      .clear();
            cornerIndices         .zeroize();
            objectToWorldMatrices .clear();
            worldAimingPoints     .clear();
            worldCameraPositions  .clear();
            worldCameraUpvectors  .clear();

            worldScreenMesh       .clear();
            worldScreenMeshAABB   .clear();
            worldAimingPoint      .zeroize();
            worldCameraPosition   .zeroize();
            worldCameraUpvector   .zeroize();
        }

        void computeFourCorners()
        {
            const int n = objectScreenMesh.numVertices();
            if( n == 0 ) { return; }

            const Vector c = objectScreenMesh.center();

            vector< pair<double,int> > list;
            {
                list.reserve( n );

                const VectorArray& p = objectScreenMesh.p;

                for( int i=0; i<n; ++i )
                {
                    list.push_back( make_pair( c.squaredDistanceTo( p[i] ), i ) );
                }
            }

            sort( list.begin(), list.end() );    // in increasing order
            reverse( list.begin(), list.end() ); // in decreasing order

            cornerIndices[0] = list[0].second;
            cornerIndices[1] = list[1].second;
            cornerIndices[2] = list[2].second;
            cornerIndices[3] = list[3].second;
        }

        Vector objectCornerPoint( const int& i /* i = 0, 1, 2, 3 */ ) const
        {
            return objectScreenMesh.p[ cornerIndices[i] ];
        }

        Vector worldCornerPoint( const int& i /* i = 0, 1, 2, 3 */ ) const
        {
            return worldScreenMesh.p[ cornerIndices[i] ];
        }

        double uMin() const
        {
            const VectorArray& uv = objectScreenMesh.uv;

            const int n = uv.length();
            if( n == 0 ) { return 0.0; }

            double min = INFINITE;

            for( int i=0; i<n; ++i )
            {
                min = MIN( min, uv[i].x );
            }

            return min;
        }

        double uMax() const
        {
            const VectorArray& uv = objectScreenMesh.uv;

            const int n = uv.length();
            if( n == 0 ) { return 0.0; }

            double max = -INFINITE;

            for( int i=0; i<n; ++i )
            {
                max = MAX( max, uv[i].x );
            }

            return max;
        }

        double vMin() const
        {
            const VectorArray& uv = objectScreenMesh.uv;

            const int n = uv.length();
            if( n == 0 ) { return 0.0; }

            double min = INFINITE;

            for( int i=0; i<n; ++i )
            {
                min = MIN( min, uv[i].y );
            }

            return min;
        }

        double vMax() const
        {
            const VectorArray& uv = objectScreenMesh.uv;

            const int n = uv.length();
            if( n == 0 ) { return 0.0; }

            double max = -INFINITE;

            for( int i=0; i<n; ++i )
            {
                max = MAX( max, uv[i].y );
            }

            return max;
        }

        void setDrawingData( const int frame /* i = 0, 1, 2, ... */ )
        {
            const Matrix& m = objectToWorldMatrix = objectToWorldMatrices[frame];

            worldScreenMesh = objectScreenMesh;
            worldScreenMesh.transform( m );

            worldScreenMeshAABB = worldScreenMesh.boundingBox();

            worldAimingPoint    = worldAimingPoints    [frame];
            worldCameraPosition = worldCameraPositions [frame];
            worldCameraUpvector = worldCameraUpvectors [frame];
        }

        Float2 worldToST( const Vector& wp ) const
        {
            // given data after executing setDrawingData()
            // : worldAimingPoint, worldCameraPosition, worldCameraUpvector

            Vector zAxis = Normalize( worldAimingPoint - worldCameraPosition );
            Vector xAxis = Normalize( Cross( zAxis, worldCameraUpvector ) );
            Vector yAxis = Normalize( Cross( xAxis, zAxis ) );

            const Vector& fPOS = wp; // why fPOS? Just like GLSL shader code.
            const Vector direction = Normalize( fPOS - worldCameraPosition );

            double xValue = Dot( direction, xAxis );
            double yValue = Dot( direction, yAxis );
            double zValue = Dot( direction, zAxis );

            Vector projectedDirection = ( xValue * xAxis ) + ( yValue * yAxis );

            double alpha = 1.0 - acos( Length( projectedDirection ) ) / ( 0.5 * M_PI );

            projectedDirection = Normalize( projectedDirection );
            projectedDirection *= alpha;

            const double x = Dot( projectedDirection, xAxis );
            const double y = Dot( projectedDirection, yAxis );

            Float2 st;
            st[0] = 0.5 * x + 0.5;
            st[1] = 0.5 * y + 0.5;

            return st;
        }

        bool save( const char* filePathName ) const
        {
            ofstream fout( filePathName, ios::out|ios::binary|ios::trunc );

            if( fout.fail() || !fout.is_open() )
            {
                cout << "Error@Manager::save(): Failed to save file: " << filePathName << endl;
                return false;
            }

            const String identifier( "BeyondScreenInfoCache" );
            identifier.write( fout );

            animationFrames       .write( fout );
            objectScreenMesh      .write( fout );
            cornerIndices         .write( fout );
            objectToWorldMatrices .write( fout );
            worldAimingPoints     .write( fout );
            worldCameraPositions  .write( fout );
            worldCameraUpvectors  .write( fout );

            fout.close();

            return true;
        }

        bool load( const char* filePathName )
        {
            Manager::clear();

            ifstream fin( filePathName, ios::in|ios::binary );

            if( fin.fail() )
            {
                cout << "Error@Manager::load(): Failed to load file." << endl;
                return false;
            }

            String identifier;
            identifier.read( fin );

            if( identifier != "BeyondScreenInfoCache" )
            {
                cout << "Error@Manager::load(): Invalid cache file." << endl;
                return false;
            }

            animationFrames       .read( fin );
            objectScreenMesh      .read( fin );
            cornerIndices         .read( fin );
            objectToWorldMatrices .read( fin );
            worldAimingPoints     .read( fin );
            worldCameraPositions  .read( fin );
            worldCameraUpvectors  .read( fin );

            fin.close();

            Manager::setDrawingData( 0 );
            hashGrid.initialize( worldScreenMesh );

            return true;
        }
};

BS_NAMESPACE_END

#endif

