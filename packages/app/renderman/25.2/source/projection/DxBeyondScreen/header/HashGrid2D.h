#ifndef _BS_HashGrid2D_h_
#define _BS_HashGrid2D_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class HashGrid2D
{
    private:

        int     _nx, _ny;      // resolution
        double  _lx, _ly;      // dimension
        double  _dx, _dy;      // cell size

        Vector _minPt, _maxPt; // two corner points of AABB

        std::vector<IntArray> _data;

    public:

        HashGrid2D()
        {
            // nothing to do
        }

        void clear()
        {
            _nx = _ny = 0;
            _lx = _ly = 0.0;
            _dx = _dy = 0.0;

            _minPt.zeroize();
            _maxPt.zeroize();

            _data.clear();
        }

        int nx() const
        {
            return _nx;
        }

        int ny() const
        {
            return _ny;
        }

        double lx() const
        {
            return _lx;
        }

        double ly() const
        {
            return _ly;
        }

        double dx() const
        {
            return _dx;
        }

        double dy() const
        {
            return _dy;
        }

        Vector minPoint() const
        {
            return _minPt;
        }

        Vector maxPoint() const
        {
            return _maxPt;
        }

        int cell( int i, int j ) const
        {
            return (i+_nx*j);
        }

        void initialize( const ScreenMesh& mesh )
        {
            const int nVertices  = mesh.numVertices();
            const int nTriangles = mesh.numTriangles();

            if( ( nVertices * nTriangles ) == 0 )
            {
                HashGrid2D::clear();
                return;
            }

            BoundingBox aabb;

            double triangleBBoxWidthSum  = 0.0;
            double triangleBBoxHeightSum = 0.0;
            {
                for( int iTri=0; iTri<nTriangles; ++iTri )
                {
                    BoundingBox triangleBBox;

                    const unsigned int& v0 = mesh.t[3*iTri  ];
                    const unsigned int& v1 = mesh.t[3*iTri+1];
                    const unsigned int& v2 = mesh.t[3*iTri+2];

                    const Vector& uv0 = mesh.uv[v0];
                    const Vector& uv1 = mesh.uv[v1];
                    const Vector& uv2 = mesh.uv[v2];

                    aabb.expand( uv0 );
                    aabb.expand( uv1 );
                    aabb.expand( uv2 );

                    triangleBBox.expand( uv0 );
                    triangleBBox.expand( uv1 );
                    triangleBBox.expand( uv2 );

                    triangleBBoxWidthSum  += triangleBBox.width(0);
                    triangleBBoxHeightSum += triangleBBox.width(1);
                }
            }

            _minPt = aabb.min();
            _maxPt = aabb.max();
            {
                const int& Nx = _nx = int( ( _maxPt.x -  _minPt.x ) / ( triangleBBoxWidthSum  / nTriangles ) )+1;
                const int& Ny = _ny = int( ( _maxPt.y -  _minPt.y ) / ( triangleBBoxHeightSum / nTriangles ) )+1;

                const double& Lx = _lx = _maxPt.x - _minPt.x;
                const double& Ly = _ly = _maxPt.y - _minPt.y;

                const double& Dx = _dx = Lx / double(Nx);
                const double& Dy = _dy = Ly / double(Ny);
            }

            _data.resize( _nx * _ny );

            for( int iTri=0; iTri<nTriangles; ++iTri )
            {
                const unsigned int& v0 = mesh.t[3*iTri  ];
                const unsigned int& v1 = mesh.t[3*iTri+1];
                const unsigned int& v2 = mesh.t[3*iTri+2];

                const Vector& p0 = mesh.uv[v0];
                const Vector& p1 = mesh.uv[v1];
                const Vector& p2 = mesh.uv[v2];

                BoundingBox box;
                box.expand( p0 );
                box.expand( p1 );
                box.expand( p2 );

                const Vector& minPt = box.min();
                const Vector& maxPt = box.max();

                const int i0 = Clamp( int((minPt.x - _minPt.x)/_dx),   0, _nx-1 );
                const int i1 = Clamp( int((maxPt.x - _minPt.x)/_dx)+1, 0, _nx-1 );
                const int j0 = Clamp( int((minPt.y - _minPt.y)/_dy),   0, _ny-1 );
                const int j1 = Clamp( int((maxPt.y - _minPt.y)/_dy)+1, 0, _ny-1 );

                for( int j=j0; j<=j1; ++j )
                for( int i=i0; i<=i1; ++i )
                {{
                    _data[ HashGrid2D::cell(i,j) ].push_back( iTri );
                }}
            }
        }

        const IntArray& candidates( double x, double y ) const
        {
            const int i = Clamp( int( ( x - _minPt.x ) / _dx ), 0, _nx-1 );
            const int j = Clamp( int( ( y - _minPt.y ) / _dy ), 0, _ny-1 );

            return _data[ HashGrid2D::cell(i,j) ];
        }
};

BS_NAMESPACE_END

#endif
