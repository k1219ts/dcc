#ifndef _BS_BoundingBox_h_
#define _BS_BoundingBox_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class BoundingBox
{
    public:

        bool _initialized;

        Vector _min;
        Vector _max;

    public:

        BoundingBox()
        : _initialized(false)
        {}

        BoundingBox( const BoundingBox& b )
        {
            BoundingBox::operator=( b );
        }

        void clear()
        {
            _initialized = false;
            _min.zeroize();
            _max.zeroize();
        }

        BoundingBox& operator=( const BoundingBox& b )
        {
            _initialized = b._initialized;
            _min = b._min;
            _max = b._max;
            return (*this);
        }

        BoundingBox& set( const Vector& p, const Vector& q )
        {
            _initialized = true;
            GetMinMax( p.x, q.x, _min.x, _max.x );
            GetMinMax( p.y, q.y, _min.y, _max.y );
            GetMinMax( p.z, q.z, _min.z, _max.z );
            return (*this);
        }

        BoundingBox& expand( const Vector& p )
        {
            if( _initialized )
            {
                _min.x = MIN( _min.x, p.x );
                _min.y = MIN( _min.y, p.y );
                _min.z = MIN( _min.z, p.z );

                _max.x = MAX( _max.x, p.x );
                _max.y = MAX( _max.y, p.y );
                _max.z = MAX( _max.z, p.z );
            }
            else
            {
                _initialized = true;
                _min = _max = p;
            }

            return (*this);
        }

        BoundingBox& expand( float epsilon=EPSILON )
        {
            if( _initialized )
            {
                _min.x -= epsilon;
                _min.y -= epsilon;

                _max.x += epsilon;
                _max.y += epsilon;

            }

            return (*this);
        }

        bool contains( const Vector& p ) const
        {
            if( !_initialized ) { return false; }

            if( p.x < _min.x ) { return false; }
            if( p.y < _min.y ) { return false; }
            if( p.z < _min.z ) { return false; }

            if( p.x > _max.x ) { return false; }
            if( p.y > _max.y ) { return false; }
            if( p.z > _max.z ) { return false; }

            return true;
        }

        bool initialized() const
        {
            return ( _initialized ? true : false );
        }

        const Vector& min() const
        {
            return _min;
        }

        const Vector& max() const
        {
            return _max;
        }

        Vector center() const
        {
            return 0.5 * ( _min + _max );
        }

        double width( int dim ) const
        {
            switch( dim )
            {
                default:
                case 0: { return ( _max.x - _min.x ); }
                case 1: { return ( _max.y - _min.y ); }
                case 2: { return ( _max.z - _min.z ); }
            }
        }

        void write( ofstream& fout ) const
        {
            fout.write( (char*)&_initialized, sizeof(int) );
            _min.write( fout );
            _max.write( fout );
        }

        void read( ifstream& fin )
        {
            fin.read( (char*)&_initialized, sizeof(int) );
            _min.read( fin );
            _max.read( fin );
        }
};

inline ostream& operator<<( ostream& os, const BoundingBox& b )
{
    os << b.min() << " ~ " << b.max() << std::endl;
    os << b.width(0) << " x " << b.width(1) << " x " << b.width(2) << endl;
	return os;
}

BS_NAMESPACE_END

#endif
